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
import os

log = 'attention_256_0.001.txt'
bs = 256
l_r = 0.001
model_save = 'attention_256_0.001.pkl'
model_para = 'attention_256_0.001_params.pkl'

embedding_dim = 200
hidden_dim = 200


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类，继承torch.utils.data.Dataset
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(root + datatxt, 'r')
        imgs = []  # 创建一个名为img的空列表，装feature
        for line in fh:
            line = line.rstrip()
            print(line)
            words = line.split()
            print(words)
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label  # return回哪些内容，训练时循环读取每个batch时，就获得哪些内容

    def __len__(self):  # 返回数据集的长度，要和loader的长度区分
        return len(self.imgs)


def loadtraindata():  # 创建数据集
    print("!!!!!!!!!!inside loadtraindata!!!!!!!!!!!!!!!")
    # path = r"/mnt/nas/cv_data/imagequality/waterloo_de20_all/train"
    # path = r"dataset/train"
    # trainset = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([transforms.Resize((32, 32)),
    #                                                                                 transforms.CenterCrop(32),
    #                                                                                 transforms.ToTensor()]))
    root = r"../"
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
    root = r"../"
    test_transformations = transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224),
                                               transforms.ToTensor()])
    test_data = MyDataset(root=root, datatxt='testset.txt', transform=test_transformations)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=True)
    return test_loader


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = nn.functional.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class AttnClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        # super().__init__()
        super(AttnClassifier, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
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
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(6400, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 50)
        )

    def set_embedding(self, vectors):
        self.embedding.weight.data.copy_(vectors)

    def forward(self, inputs, lengths):
        feature_out = self.feature(inputs)
        res = feature_out.view(feature_out.size(0), -1)
        print("res" + str(res.shape))
        out = self.dense(res)
        batch_size = out.size(1)
        # (L, B)
        embedded = self.embedding(out)
        # (L, B, E)
        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        out, hidden = self.lstm(packed_emb)
        out = nn.utils.rnn.pad_packed_sequence(out)[0]
        out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        # (L, B, H)
        embedding, attn_weights = self.attention(out.transpose(0, 1))
        # (B, HOP, H)
        outputs = self.fc(embedding.view(batch_size, -1))
        # (B, 1)
        return outputs, attn_weights


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
        # 一个错误：RuntimeError: size mismatch, m1: [4 x 6400], m2: [9216 x 4096]
        # All you have to care is b = c and you are done: m1: [a x b], m2: [c x d]
        # m1 is [a x b] which is [batch size x in features]
        # in features不是输入图像大小，输入图像为96*96时为256，224*224时为6400，227*227时为9216
        # m2 is [c x d] which is [ in features x out features]
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(6400, 4096),
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
    # optimizer = optim.Adam(net.parameters(), lr=l_r, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer = optim.SGD(net.parameters(), lr=l_r, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net.to(device)
    # train
    for epoch in range(2):
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
    # 保存模型
    torch.save(net, model_save)
    torch.save(net.state_dict(), model_para)


def reload_net():
    trainednet = torch.load(model_save)
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
