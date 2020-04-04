# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data

log = '5_4_0.001.txt'
bs = 4
l_r = 0.001


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, root, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            print(line)
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            print(words)
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定

        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片

        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


# 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
def loadtraindata():
    print("!!!!!!!!!!inside loadtraindata!!!!!!!!!!!!!!!")
    # path = r"/mnt/nas/cv_data/imagequality/waterloo_de20_all/train"
    # path = r"dataset/train"
    # trainset = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([transforms.Resize((32, 32)),
    #                                                                                 transforms.CenterCrop(32),
    #                                                                                 transforms.ToTensor()]))
    root = r"./"
    # train_transformations = transforms.Compose(
    #                         [transforms.RandomHorizontalFlip(),  # transforms.RandomCrop(32, padding=4),
    #                          transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_transformations = transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224),
                                                transforms.ToTensor()])
    train_data = MyDataset(root=root, datatxt='trainset.txt', transform=train_transformations)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)

    return train_loader


def loadtestdata():
    print("!!!!!!!!!!inside loadtestdata!!!!!!!!!!!!!!!")
    f1 = open(log, 'r+')
    f1.read()
    f1.write("\n!!!!!!!!!!inside loadtestdata!!!!!!!!!!!!!!!\n")
    f1.close()
    root = r"./"
    test_transformations = transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224),
                                               transforms.ToTensor()])
    test_data = MyDataset(root=root, datatxt='testset.txt', transform=test_transformations)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=True)
    return test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),  # output_size = 27*27*96
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),  # output_size = 13*13*256
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)  # output_size = 6*6*256
        )

        # 网络前向传播过程
        # RuntimeError: size mismatch, m1: [1000 x 6400], m2: [9216 x 4096]
        # All you have to care is b = c and you are done: m1: [a x b], m2: [c x d]
        # m1 is [a x b] which is [batch size x in features] in features不是输入图像大小，输入图像为96*96时为256，输入图像为227*227时为9216
        # m2 is [c x d] which is [ in features x out features]
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 50)
        )

    def forward(self, x):
        feature_out = self.feature(x)
        res = feature_out.view(feature_out.size(0), -1)
        print("res" + str(res.shape))
        out = self.dense(res)
        return out


classes = ('stay', 'left', 'up', 'right', 'down')


def trainandsave():
    print("!!!!!!!!!!inside trainandsave!!!!!!!!!!!!!!!")
    trainloader = loadtraindata()

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=l_r, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 假设我们在支持CUDA的机器上，我们可以打印出CUDA设备：
    print(device)

    net.to(device)
    # train
    for epoch in range(3):
        running_loss = 0.0
        print("\n-----------------------------------\nepoch " + str(epoch) + ":")
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            print("i:" + str(i))
            inputs, labels = data
            print("input:")
            print(inputs)
            print("labels:")
            print(labels)

            # wrap them in Variable
            # inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # running_loss += loss.data[0]
            running_loss += loss.item()
            print("outputs:")
            print(outputs)
            print("loss:")
            print(loss)
            print("running_loss:")
            print(running_loss)
            f1 = open(log, 'r+')
            f1.read()
            f1.write("\n-------------------------------------------\nepoch " + str(epoch) + ":")
            f1.write("i:" + str(i))
            f1.write("\nloss:")
            f1.write(str(loss))
            f1.write("\nrunning_loss:")
            f1.write(str(running_loss))
            f1.close()

            if i % 200 == 199:
                print('\n[%d, %5d] loss: %.3f' % (epoch, i, running_loss / 200))
                f1 = open(log, 'r+')
                f1.read()
                f1.write('\n[%d, %5d] loss: %.3f' % (epoch, i, running_loss / 200))
                f1.close()
                running_loss = 0.0

    print('Finished Training')
    f1 = open(log, 'r+')
    f1.read()
    f1.write('Finished Training\n')
    f1.close()
    torch.save(net, '5_4_0.001.pkl')
    torch.save(net.state_dict(), '5_4_0.001_params.pkl')


def reload_net():
    trainednet = torch.load('5_4_0.001.pkl')
    return trainednet


def test():
    print("!!!!!!!!!!inside test!!!!!!!!!!!!!!!")
    f1 = open(log, 'r+')
    f1.read()
    f1.write("\n!!!!!!!!!!inside test!!!!!!!!!!!!!!!\n")
    f1.close()

    testloader = loadtestdata()

    f1 = open(log, 'r+')
    f1.read()
    f1.write("\n!!!!!!!!!!after loadtest data!!!!!!!!!!!!!!!\n")
    f1.close()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = reload_net()
    net.to(device)
    dataiter = iter(testloader)
    images, labels = dataiter.next()  #
    # imshow(torchvision.utils.make_grid(images, nrow=5))
    images, labels = images.to(device), labels.to(device)
    print('GroundTruth: ', " ".join('%5s' % classes[labels[j]] for j in range(4)))
    f1 = open(log, 'r+')
    f1.read()
    f1.write('GroundTruth: ' + " ".join('%5s' % classes[labels[j]] for j in range(4)) + '\n')
    f1.close()

    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(4)))
    f1 = open(log, 'r+')
    f1.read()
    f1.write('Predicted: ' + " ".join('%5s' % classes[predicted[j]] for j in range(4)) + '\n')
    f1.close()

    images, labels = dataiter.next()  #
    # imshow(torchvision.utils.make_grid(images, nrow=5))
    images, labels = images.to(device), labels.to(device)
    print('GroundTruth: ', " ".join('%5s' % classes[labels[j]] for j in range(4)))
    f1 = open(log, 'r+')
    f1.read()
    f1.write('GroundTruth: ' + " ".join('%5s' % classes[labels[j]] for j in range(4)) + '\n')
    f1.close()

    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(4)))
    f1 = open(log, 'r+')
    f1.read()
    f1.write('Predicted: ' + " ".join('%5s' % classes[predicted[j]] for j in range(4)) + '\n')
    f1.close()

    # 评估测试数据集
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (labels == predicted).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    f1 = open(log, 'r+')
    f1.read()
    f1.write('\nAccuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    f1.close()

    # 按类标评估
    n_classes = len(classes)
    class_correct, class_total = [0] * n_classes, [0] * n_classes

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            is_correct = (labels == predicted).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                class_correct[label] += is_correct[i].item()

    for i in range(n_classes):
        print('Accuracy of %5s: %.2f %%' % (
            classes[i], 100.0 * class_correct[i] / class_total[i]
        ))
        f1 = open(log, 'r+')
        f1.read()
        f1.write('\nAccuracy of %5s: %.2f %%' % (
            classes[i], 100.0 * class_correct[i] / class_total[i]
        ))
        f1.close()

    # 显示一张图片


def imshow(img):
    img = img / 2 + 0.5  # 逆归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    # 任意地拿到一些图片


trainandsave()
test()
