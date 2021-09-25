import torch
import torchvision
import torchvision.transforms as transforms
import os
import util
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from distutils.util import strtobool
from argument import  parse_args
from model.resnet import resnet18
from model.lenet import LeNet
from model.resmlp import ResMLP
# from retraning import freez_param
import datetime
from model.mlp_mixier.mlp_mixier_pytorch import MLPMixer


def model_choice ():
    args = parse_args()
    saved_path = ''
    if args.model =='resnet18':
        # model= resnet_model()
        model = resnet18()
        # saved_path = 'ここにパスを指定/_best.pth'
        saved_path = 'weights/resnet18-210819_223705/_best.pth'
    if args.model =='lenet':
        model = LeNet()
        saved_path =  'weights/93/lenet99_best.pth'

    if args.model =='mixier':
        model = MLPMixer(
            image_size = 32,
            channels = 3,
            patch_size = 4,#16 for imagenet 
            dim = 64,#パッチ切り出しした後のnn.linearの出力がdim 
            depth = 9,#num_layer = 2*depth +2  9 
            num_classes = 10
        )
    if args.model =='resmlp':
        model = ResMLP(in_channels=3, image_size=32, patch_size=4, num_classes=10,
                dim=384, depth=12, mlp_dim=384*4)
    return model, saved_path

def main():
    args = parse_args()
    batch_size_train = 128
    batch_size_test = 128
    num_shown_images = 8
    input_size = 32
    # input_size = 64
    study_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    transform_train = transforms.Compose([
        torchvision.transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
    ])
    transform_test = transforms.Compose([
        torchvision.transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
    ])

    trainset = torchvision.datasets.CIFAR10(root='/work/Shared/Datasets', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='/work/Shared/Datasets', train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                            shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # num_classes = len(classes)

    net,saved_path = model_choice() 
    if args.load:
        net.load_state_dict(torch.load(saved_path))
        print(f'load weight success')


    net.to(args.device)
    
    train(net,study_name,trainloader,testloader)

        # # 最適な成果を取得
    # best_trial = study.best_trial

    # その時のパラメータと学習済み係数でモデルを復元
    # net = Net(optuna.trial.FixedTrial(best_trial.params))


def train(net,study_name,trainloader,testloader):
    args = parse_args()
    device = args.device 
    # date_str = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # basename = "%s-%s" % (study_name, date_str)
    basename = "%s-%s" %(args.model,study_name)
    
    print("Starting training for '%s'" % basename)
   
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer = util.get_optimizer(net, args.optimizer, args.lr,args.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)

    dataiter = iter(trainloader)
    images, _ = dataiter.next()
    writer = SummaryWriter("runs/0817/%s" % basename)
    writer.add_graph(net, images.to(device))

    criterion = nn.CrossEntropyLoss()

    # Save initial state
    util.add_param(writer, net, 0)
    acc_max =0.

    # try:
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        start_time = time.time()
        running_loss = 0.0
        net.train()

        

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:
                train_acc = util.accuracy_batch(outputs, labels)
                print('[%d, %5d] loss: %.3f, train batch acc: %2d %%' %
                    (epoch + 1, i + 1, running_loss, train_acc))

                gstep = epoch * len(trainloader) + i
                #fixme : tensor boardでは横軸に小数を取ってくるのができない（？）ため、横軸にgstepを持ってきています。
                writer.add_scalar('Training/Loss', running_loss, gstep)
                writer.add_scalar('Training/Accuracy', train_acc, gstep)

                running_loss = 0.0

        scheduler.step()

        # Evaluate intermediate result
        gstep = epoch * len(trainloader) + i
        net.eval()
        with util.IntermediateOutputWriter(writer, net, gstep):
            test_acc = util.accuracy(testloader, net, device=device)
            print('[%d,      ] test acc: %.3f %%' %
                (epoch + 1, test_acc))

        writer.add_scalar('Test/Accuracy', test_acc, epoch+1)
        writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], epoch+1)

        util.add_param(writer, net, epoch+1 )
        end_time = time.time()
        print('{:.1f}'.format(end_time-start_time))

        is_best = (test_acc > acc_max) 
        if is_best:
            if args.save_weight:
                acc_max = max(acc_max, test_acc)                
                save_weight_path = util.save_weight(net,basename)
                print('saved best weight:  %s'%save_weight_path)


if __name__ == '__main__':
    main()
