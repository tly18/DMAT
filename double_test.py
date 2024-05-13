import torch.optim as optim
import torch.utils.data
from torchvision.datasets import mnist, CIFAR10
import torchvision
from torchvision import models
from torchvision import transforms as transforms
import torch
import torch.nn as nn
from models.resnet_test import ResNet18

import numpy as np
import os
from advertorch.attacks import LinfPGDAttack,CarliniWagnerL2Attack,DDNL2Attack,SpatialTransformAttack
from torch.autograd import Variable
import torch.nn.functional as F
import torchattacks
from autoattack import AutoAttack
import time
from MIFGSM import MIFGSM

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

def pgd_whitebox_attack_test(model_1,model_2,
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
            loss = nn.CrossEntropyLoss()(model_1(X_pgd), y) + nn.CrossEntropyLoss()(model_2(X_pgd), y)
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

class double_net_attacker(torch.nn.Module):
    def __init__(self,ATnetwork=None,ATnetwork_2=None,device='cuda:0', beta=0.5) -> None:
        super().__init__()
        self.ATnetwork=ATnetwork
        self.ATnetwork_2 = ATnetwork_2
        self.device = device
        self.beta = beta
        
    def forward(self, x):
        out= self.ATnetwork(x)
        out_2 = self.ATnetwork_2(x)
        out_final = ((1-self.beta) * F.softmax(out, dim=1) + self.beta * F.softmax(out_2, dim=1)) / 2.0
        return out_final

def CW( model_1, model_2, img, gt, margin, epsilon, device, num_classes, num_steps,step_size ):
    model_1.eval()
    model_2.eval()
    x_adv = Variable(img.data, requires_grad=True)
    random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-epsilon, epsilon).to(device)
    x_adv = Variable(x_adv.data + random_noise, requires_grad=True)
    onehot_targets = torch.eye(num_classes)[gt].to(device)

    for _ in range(num_steps):
        opt = optim.SGD([x_adv], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            logits = model_1(x_adv) + model_2(x_adv)

            self_loss = torch.sum(onehot_targets * logits, dim=1)
            other_loss = torch.max((1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

            loss = -torch.sum(torch.clamp(self_loss - other_loss + margin, 0))
            loss = loss / onehot_targets.shape[0]

        loss.backward()
        eta = step_size * x_adv.grad.data.sign()
        x_adv = Variable(x_adv.data + eta, requires_grad=True)
        eta = torch.clamp(x_adv.data - img.data, -epsilon, epsilon)
        x_adv = Variable(img.data + eta, requires_grad=True)
        x_adv = Variable(torch.clamp(x_adv, 0, 1.0), requires_grad=True)
    return x_adv

if __name__ == '__main__':
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    #CIFAR10 train_dataset&train_dataset
    train_batch_size = 128
    test_batch_size = 128
    # transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # ])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])
    # train_set = CIFAR10(root='./data/', train=True, download=False, transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, num_workers=10)
    test_set = CIFAR10(root='./data/', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False, num_workers=10)


    #对抗训练
    # model = VGG16().to(device)
    # model = WideResNet(depth=34, num_classes=10).to(device) 
    # model = ResNet(18, 10)
    # model.to(device)
    # model = models.resnet18(pretrained=False).to(device)

    model_1 = ResNet18().to(device)
    model_2 = ResNet18().to(device)
    model_1.load_state_dict(torch.load('./para/double/cri/pgd_AT_double_resnet18_cifar10/lr_3_5e_3/2/model_1_93_standard_pgd_AT_cifar10.pt', map_location=device))
    model_2.load_state_dict(torch.load('./para/double/cri/pgd_AT_double_resnet18_cifar10/lr_3_5e_3/2/model_2_93_standard_pgd_AT_cifar10.pt', map_location=device))
    # model_1.load_state_dict(torch.load('./para/double/line2/model_1_97_standard_pgd_AT_cifar10.pt', map_location=device))
    # model_2.load_state_dict(torch.load('./para/double/line2/model_2_97_standard_pgd_AT_cifar10.pt', map_location=device))
    
    # mnist
    # model_1.load_state_dict(torch.load('./MNIST/para/double_resnet/cri/pgd_AT_double_resnet18_cifar10/lr_3_5e_3/2/model_2_71_standard_pgd_AT_cifar10.pt', map_location=device))
    # model_2.load_state_dict(torch.load('./MNIST/para/double_resnet/cri/pgd_AT_double_resnet18_cifar10/lr_3_5e_3/2/model_1_71_standard_pgd_AT_cifar10.pt', map_location=device))


    # model.load_state_dict(torch.load('./para/AT_resnet18_fgsm/119_standard_AT_cifar10_resnet18.pt', map_location=device))
    # torch.load(os.path.join('./para/AT_resnet18_fgsm/',str(epoch)+'119_standard_AT_cifar10_resnet18.pt'), map_location={'cuda:0':'cuda:2'})
    
    # beta = nn.Parameter(torch.randn(1, 1).to(device))
    beta = torch.nn.Parameter(torch.tensor([0.5302348732948303]).to(device), requires_grad=True)      #93
    # beta = torch.nn.Parameter(torch.tensor([0.36818382143974304]).to(device), requires_grad=True)       # 97
    print('93')
    
    # optimizer_beta = optim.SGD([beta], lr=0.5)
    
    criterion = nn.CrossEntropyLoss().to(device)
    kl = nn.KLDivLoss(reduction='none').to(device)      #reduction='sum' size_average=False batchmean
    criterion_kl = nn.KLDivLoss(size_average=False).to(device)
 
    for i in range(0,1,1):       #  (120,epoch,1)   epoch
       
        
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
                output_1 = model_1(data.float())  # 预测输出
                output_2 = model_2(data.float())
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


            # 生成对抗样本
            
            #aa
            # integration_net = double_net_attacker(model_1,model_2,device=device,beta=beta)
            # adversary = AutoAttack(integration_net, norm='Linf', eps=8/255, device=device, verbose = False) 
            # data_adv = adversary.run_standard_evaluation(data.float(), target, bs=128).to(device)
            
            # cw           # model_1, model_2, img, gt, margin, epsilon, device, num_classes, num_steps,step_size
            data_adv = CW( model_1, model_2, data.float(), target, 1, 8/255, device, 10, 10, 0.007)
            
            # mim
            # data_adv = MIFGSM(model_1, model_2, data, target, eps=8/255, alpha=2/255, steps=10, decay=1.0,device=device)
            
            # pgd-20
            # data_adv = pgd_whitebox_attack(model_1,model_2,data,target,epsilon=0.031,num_steps=20,step_size=0.003).to(device)
            
            # pgd-100
            # data_adv = pgd_whitebox_attack_test(model_1,model_2,data,target,epsilon=0.031,num_steps=100,step_size=0.0007).to(device)
                        
            with torch.no_grad():
                output_adv_1 = model_1(data_adv.float())  # 预测输出
                output_adv_2 = model_2(data_adv.float())

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
        print(beta, 'nat correct 1:', test_nat_correct_1 / test_total, 'nat correct 2:', test_nat_correct_2 / test_total, 'nat correct 3:', test_nat_correct_3 / test_total, 'nat correct 4:', test_nat_correct_4 / test_total)
        print(beta,   'adv correct 1:', test_adv_correct_1 / test_total, 'adv correct 2:', test_adv_correct_2 / test_total, 'adv correct 3:', test_adv_correct_3 / test_total, 'adv correct 4:', test_adv_correct_4 / test_total,'\n')


