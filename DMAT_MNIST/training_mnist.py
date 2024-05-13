import torch.optim as optim
import torch.utils.data
from torchvision.datasets import mnist, CIFAR10
import torchvision
from torchvision import models
from torchvision import transforms as transforms
import torch
import torch.nn as nn
from lenet import lenet
from res_mnist import ResNet18
import numpy as np
import os
from advertorch.attacks import LinfPGDAttack,CarliniWagnerL2Attack,DDNL2Attack,SpatialTransformAttack
from torch.autograd import Variable
import torch.nn.functional as F
import torchattacks
from autoattack import AutoAttack
import time

def cosine_similarity(a, b):
    batch, dimen = a.shape
    cosine_sim_loss = 0
    for i in range(0,batch,1):
        dot_product = torch.dot(a[i,:].view(dimen), b[i,:].view(dimen))
        norm_a = torch.norm(a)
        norm_b = torch.norm(b)
        cosine_sim = dot_product / (norm_a * norm_b)
        cosine_sim_loss = cosine_sim_loss + cosine_sim
    return cosine_sim_loss/batch

# generate adversarial example
def generation_adv(model_1,model_2,
              x_natural,
              y,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              distance='l_inf'):                                                                                                                                     
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model_1(x_adv)[0], y) + F.cross_entropy(model_2(x_adv)[0], y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv

def pgd_whitebox_attack(model_1,model_2,
                  X,
                  y,
                  epsilon=0.031,
                  num_steps=20,
                  step_size=0.003):

    X_pgd = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_1(X_pgd)[0], y) + nn.CrossEntropyLoss()(model_2(X_pgd)[0], y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def calculate_mart_loss(kl, train_batch_size, output_adv, output_nat):

    adv_probs = F.softmax(output_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(output_adv, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(output_nat, dim=1)
    true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / train_batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs) *\
        (0.0000001 + output_nat.data.max(1)[1] != target.data).float())
    loss = loss_adv + float(6) * loss_robust
    return loss

def calculate_trades_loss(criterion_kl, train_batch_size, output_adv, output_nat, loss_nat, beta=6.0):
    loss_robust = (1.0 / train_batch_size) * criterion_kl(F.log_softmax(output_adv, dim=1),
                                                    F.softmax(output_nat, dim=1))
    loss = loss_nat + beta * loss_robust
    return loss
    
def line_loss(result, target):

    batch, dimen = result.shape 
    loss = Variable(torch.as_tensor(0,dtype=torch.float),requires_grad=True).to(device)
    
    mask = result < 0
    result_n = torch.FloatTensor(result.size()).type_as(result)
    result_n[mask] = torch.exp(result[mask])
    mask = result >= 0
    result_n[mask] = torch.add(result[mask], 1)
    
    # result_n = torch.exp(result)

    line_sum = torch.sum(result_n,dim=1) + 0.00001
    line_sum=torch.reshape(line_sum,[batch,1])
    result_a = result_n / line_sum.repeat(1,dimen)
    result_loss = torch.log(result_a + 0.00001)

    loss_NL=torch.nn.NLLLoss()
    loss = loss_NL(result_loss,target)
    return loss

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # train_dataset&train_dataset
    train_batch_size = 128
    test_batch_size = 128
    train_dataset = mnist.MNIST(root='./dataset', train=True, download=False, transform=transforms.ToTensor())
    test_dataset = mnist.MNIST(root='./dataset', train=False, download=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=10)


    #对抗训练
    # model = VGG16().to(device)
    # model = WideResNet(depth=34, num_classes=10).to(device) 
    # model = ResNet(18, 10)
    # model.to(device)
    # model = models.resnet18(pretrained=False).to(device)

    model_1 = ResNet18().to(device)
    model_2 = ResNet18().to(device)
    # model_1.load_state_dict(torch.load('./para/line2024/line_pgd_AT_double_resnet18_cifar10/model_1_71_standard_pgd_AT_cifar10.pt', map_location=device))
    # model_2.load_state_dict(torch.load('./para/line2024/line_pgd_AT_double_resnet18_cifar10/model_2_71_standard_pgd_AT_cifar10.pt', map_location=device))

    # model.load_state_dict(torch.load('./para/AT_resnet18_fgsm/119_standard_AT_cifar10_resnet18.pt', map_location=device))
    # torch.load(os.path.join('./para/AT_resnet18_fgsm/',str(epoch)+'119_standard_AT_cifar10_resnet18.pt'), map_location={'cuda:0':'cuda:2'})

    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer_1 = optim.SGD(model_1.parameters(), lr=0.01, momentum=0.9, weight_decay=3.5e-3)     # 1e-3
    optimizer_2 = optim.SGD(model_2.parameters(), lr=0.01, momentum=0.9, weight_decay=3.5e-3)
    
    # beta = nn.Parameter(torch.randn(1, 1).to(device))
    beta = torch.nn.Parameter(torch.tensor([0.5]).to(device), requires_grad=True)
    # optimizer_beta = optim.SGD([beta], lr=0.5)
    
    scheduler_1 = optim.lr_scheduler.MultiStepLR(optimizer_1, milestones=[75, 90, 110], gamma=0.1)
    scheduler_2 = optim.lr_scheduler.MultiStepLR(optimizer_2, milestones=[75, 90, 110], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    kl = nn.KLDivLoss(reduction='none').to(device)      #reduction='sum' size_average=False batchmean
    criterion_kl = nn.KLDivLoss(size_average=False).to(device)

    epoch = 120
    flag = 0
    flag_epoch = 0

    pth_save_dir = os.path.join('./para/double_resnet/cri/pgd_AT_double_resnet18_cifar10/lr_3_5e_3/2/')
    if not os.path.exists(pth_save_dir):
        os.makedirs(pth_save_dir)
 
    for _epoch in range(epoch):       #  (120,epoch,1)   epoch
        
        # 训练
        print("train:")
        model_1.train()
        model_2.train()
        train_loss = 0
        train_adv_loss = 0
        train_nat_correct_1 = 0
        train_adv_correct_1 = 0
        train_nat_correct_2 = 0
        train_adv_correct_2 = 0
        train_total = 0
        # 一次epoch
        t0 = time.time()
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            
            optimizer_1.zero_grad()  # 梯度归零   
            optimizer_2.zero_grad()
            # optimizer_beta.zero_grad()
            
            beta.data = torch.clamp(beta.data, min=0, max=1)
           
            model_1.eval()
            model_2.eval()
            # 执行攻击 generate adversarial examples
            
            # PGD
            # adversary = LinfPGDAttack(
            #     model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.031,
            #     nb_iter=10, eps_iter=0.007, rand_init=True, clip_min=0.0,
            #     clip_max=1.0, targeted=False)
            # data_adv = adversary.perturb(data.float(), target).to(device)
            data_adv = generation_adv(model_1,model_2,data,target,step_size=0.07,epsilon=0.3,perturb_steps=10,distance='l_inf')
            
            # FGSM
            # adversary = torchattacks.FGSM(model, eps=8/255)
            # data_adv = adversary(data, target)
          
            data_adv.to(device)
            model_1.train()
            model_2.train()
      
            optimizer_1.zero_grad()  # 梯度归零   
            optimizer_2.zero_grad()
            # optimizer_beta.zero_grad()
            

            output_nat_1, output_nat_1_c = model_1(data.float())      # 预测输出
            output_adv_1, output_adv_1_c = model_1(data_adv.float())  # 预测输出
            output_nat_2, output_nat_2_c = model_2(data.float())      # 预测输出
            output_adv_2, output_adv_2_c = model_2(data_adv.float())  # 预测输出
            
            loss_nat_1 = criterion(output_nat_1, target)  # 交叉熵损失
            loss_adv_1 = criterion(output_adv_1, target)  # 交叉熵损失
            loss_nat_2 = criterion(output_nat_2, target)  # 交叉熵损失
            loss_adv_2 = criterion(output_adv_2, target)  # 交叉熵损失
            
            # loss_nat_1 = line_loss(output_nat_1, target)  
            # loss_adv_1 = line_loss(output_adv_1, target)  
            # loss_nat_2 = line_loss(output_nat_2, target)  
            # loss_adv_2 = line_loss(output_adv_2, target)  

            # loss_adv_1 = calculate_mart_loss(kl, train_batch_size, output_adv_1, output_nat_1)
            # loss_adv_2 = calculate_mart_loss(kl, train_batch_size, output_adv_2, output_nat_2)
            
            # loss_adv_1 = calculate_trades_loss(criterion_kl, train_batch_size, output_adv_1, output_nat_1, loss_nat_1, beta=6.0)
            # loss_adv_2 = calculate_trades_loss(criterion_kl, train_batch_size, output_adv_2, output_nat_2, loss_nat_2, beta=6.0)
            
            loss = beta * (loss_adv_1) + (1-beta) * (loss_adv_2) + cosine_similarity(output_adv_1_c, output_adv_2_c)
            # loss = beta * (loss_adv_1 + 0.5 * loss_nat_1) + (1-beta) * (loss_adv_2 + 0.5 * loss_nat_2) + cosine_similarity(output_nat_1_c, output_nat_2_c)
            # loss = beta * loss_adv_1 + (1-beta) * loss_adv_2 + cosine_similarity(output_adv_1_c, output_adv_2_c)
            # loss = beta * (loss_adv_1 + 0.5 * loss_nat_1) + (1-beta) * (loss_adv_2 + 0.5 * loss_nat_2) + 0.5 * (cosine_similarity(output_adv_1_c, output_adv_2_c) + cosine_similarity(output_nat_1_c, output_nat_2_c))
            # loss =   loss_adv_1 +  loss_adv_2 + cosine_similarity(output_adv_1_c, output_adv_2_c)
            # print(loss_adv_1,loss_adv_2)
            
            if _epoch == 0 and batch_num == 0:
                # data是前面运行出的数据，先将其转为字符串才能写入
                result2txt = "CRI loss = beta * (loss_adv_1) + (1-beta) * (loss_adv_2) + cosine_similarity(output_adv_1_c, output_adv_2_c) "
                with open(pth_save_dir + 'pgd_AT结果存放.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
                    file_handle.write(result2txt)  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    
            torch.autograd.set_detect_anomaly(True)

            loss.backward()  # 梯度反传
            
            # 手动计算 beta 的梯度
            # beta_grad = torch.autograd.grad(loss, beta, create_graph=True)[0]
            beta_grad = loss_adv_1 - loss_adv_2

            # 使用计算得到的梯度来更新 beta 参数
            with torch.no_grad():
                beta += 0.0005 * beta_grad
            # print(loss_adv_1 , loss_adv_2,beta_grad, beta,cosine_similarity(output_adv_1_c, output_adv_2_c))
            optimizer_1.step()  # 执行一次优化步骤，通过梯度下降法来更新参数的值
            optimizer_2.step()
            # optimizer_beta.step()
            
            # print(beta.grad)      #, optimizer_beta.param_groups[0]['params']

            # train_adv_loss += loss_adv.item() # 累加损失值
            train_adv_correct_1 += torch.sum(torch.max(output_adv_1, 1)[1] == target)
            train_adv_correct_2 += torch.sum(torch.max(output_adv_2, 1)[1] == target)    
            
            # train_loss += loss_nat.item()  # 累加损失值
            train_nat_correct_1 += torch.sum(torch.max(output_nat_1, 1)[1] == target)    # dim=1指行 1位置  正确加1
            train_nat_correct_2 += torch.sum(torch.max(output_nat_2, 1)[1] == target) 
            

            train_total += target.size(0) 

        scheduler_1.step()
        scheduler_2.step()
        t1 = time.time()
        print('训练时间：', t1-t0)
        print(_epoch, '1 nat:', train_nat_correct_1 / train_total, '1 adv:', train_adv_correct_1 / train_total)
        print(beta, '2 nat:', train_nat_correct_2 / train_total, '2 adv:', train_adv_correct_2 / train_total)
 

        # 测试
        print("test:")
        model_1.eval()
        model_2.eval()
        
        test_loss = 0
        test_nat_correct_1 = 0
        test_nat_correct_2 = 0
        test_nat_correct_3 = 0
        test_nat_correct_4 = 0

        test_adv_loss = 0
        test_adv_correct_1 = 0
        test_adv_correct_2 = 0
        test_adv_correct_3 = 0
        test_adv_correct_4 = 0
        
        test_total = 0

        # 一次epoch
        # with torch.no_grad():
        t2 = time.time()
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output_1, _ = model_1(data.float())  # 预测输出
                output_2, _ = model_2(data.float())
            # loss_ = criterion(output, target)  # 交叉熵损失
            # test_loss += loss_.item()
            test_nat_correct_1 += torch.sum(torch.max(output_1, 1)[1] == target)  # 正确次数
            test_nat_correct_2 += torch.sum(torch.max(output_2, 1)[1] == target)
            
            output_3 = ((1-beta) * F.softmax(output_1, dim=1) + beta * F.softmax(output_2, dim=1)) / 2.0
            test_nat_correct_3 += torch.sum(torch.max(output_3, 1)[1] == target)
            
            output_4 = (beta * F.softmax(output_1, dim=1) + (1-beta) * F.softmax(output_2, dim=1)) / 2.0
            test_nat_correct_4 += torch.sum(torch.max(output_4, 1)[1] == target)

            # 执行攻击 generate adversarial examples
            # PGD
            # adversary = LinfPGDAttack(
            #     model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.031,
            #     nb_iter=20, eps_iter=0.003, rand_init=True, clip_min=0.0,
            #     clip_max=1.0, targeted=False)
            # data_adv_ = adversary.perturb(data.float(), target)

            # adversary = torchattacks.PGD(model, eps=0.031, alpha=0.003, steps=20, random_start=True)
            # data_adv_ = adversary(data, target).to(device)

            optimizer_1.zero_grad()  # 梯度归零   
            optimizer_2.zero_grad()
            # optimizer_beta.zero_grad()
            # 生成对抗样本
            data_adv = pgd_whitebox_attack(model_1,model_2,data,target,epsilon=0.3,num_steps=20,step_size=0.03).to(device)
            
            with torch.no_grad():
                output_adv_1, _ = model_1(data_adv.float())  # 预测输出
                output_adv_2, _ = model_2(data_adv.float())

            # _loss = criterion(output_adv, target)  # 交叉熵损失
            # test_adv_loss += _loss.item()  # 累加损失值

            test_adv_correct_1 += torch.sum(torch.max(output_adv_1, 1)[1] == target)
            test_adv_correct_2 += torch.sum(torch.max(output_adv_2, 1)[1] == target)
            
            output_adv_3 = ((1-beta) * F.softmax(output_adv_1, dim=1) + beta * F.softmax(output_adv_2, dim=1)) / 2.0
            test_adv_correct_3 += torch.sum(torch.max(output_adv_3, 1)[1] == target)
            
            output_adv_4 = (beta * F.softmax(output_adv_1, dim=1) + (1-beta) * F.softmax(output_adv_2, dim=1)) / 2.0
            test_adv_correct_4 += torch.sum(torch.max(output_adv_4, 1)[1] == target)
            
            test_total += target.size(0)

        t3 = time.time()
        print('测试时间', t3-t2)
        print(_epoch, 'nat correct 1:', test_nat_correct_1 / test_total, 'nat correct 2:', test_nat_correct_2 / test_total, 'nat correct 3:', test_nat_correct_3 / test_total, 'nat correct 4:', test_nat_correct_4 / test_total)
        print(beta,   'adv correct 1:', test_adv_correct_1 / test_total, 'adv correct 2:', test_adv_correct_2 / test_total, 'adv correct 3:', test_adv_correct_3 / test_total, 'adv correct 4:', test_adv_correct_4 / test_total,'\n')

        if _epoch >= 70:
            torch.save(model_1.state_dict(), os.path.join(pth_save_dir, 'model_1_%d_standard_pgd_AT_cifar10.pt' % _epoch))
            torch.save(model_2.state_dict(), os.path.join(pth_save_dir, 'model_2_%d_standard_pgd_AT_cifar10.pt' % _epoch))
        
        # data是前面运行出的数据，先将其转为字符串才能写入
        result2txt = str(_epoch) + ',' + str(beta.item()) + ',训练: clean:' + str(round(((train_nat_correct_1 / train_total)*100).item(),4)) + ',' + str(round(((train_nat_correct_2 / train_total)*100).item(),4)) + ',' + \
            'adv:' + str(round(((train_adv_correct_1 / train_total)*100).item(),4)) + ',' + str(round(((train_adv_correct_2 / train_total)*100).item(),4))  + ',测试: clean:' +\
            str(round(((test_nat_correct_1 / test_total)*100).item(),4)) + ',' + str(round(((test_nat_correct_2 / test_total)*100).item(),4)) + ',' + str(round(((test_nat_correct_3 / test_total)*100).item(),4)) + ',' + str(round(((test_nat_correct_4 / test_total)*100).item(),4)) + ','\
            'adv:'+ str(round(((test_adv_correct_1 / test_total)*100).item(),4)) + ',' + str(round(((test_adv_correct_2 / test_total)*100).item(),4)) + ',' + str(round(((test_adv_correct_3 / test_total)*100).item(),4)) + ',' + str(round(((test_adv_correct_4 / test_total)*100).item(),4)) + '训练时间:' + str(t1-t0) +'测试时间:' + str(t3-t2)
    
        with open(pth_save_dir + 'pgd_AT结果存放.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            file_handle.write(result2txt)  # 写入
            file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
