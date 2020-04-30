# encoding:utf-8
import torch as t
import torch.nn as nn
from torchvision import models
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os.path


net = t.load('../models/attention_256_0.001.pkl')
savedir = r'../hotmap/attention_256_0.001/'

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
        self.feature = t.nn.Sequential(
            t.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=3, stride=2),  # output_size = 27*27*96
            t.nn.Conv2d(96, 256, 5, 1, 2),
            t.nn.ReLU(),
            t.nn.MaxPool2d(3, 2),  # output_size = 13*13*256
            t.nn.Conv2d(256, 384, 3, 1, 1),
            t.nn.ReLU(),  # output_size = 13*13*384
            t.nn.Conv2d(384, 256, 3, 1, 1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(3, 2)  # output_size = 6*6*256
        )
        self.dense = t.nn.Sequential(
            t.nn.Linear(6400, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(0.5),
            t.nn.Linear(4096, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(0.5),
            t.nn.Linear(4096, 50)
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


class FeatureExtractor(nn.Module):
    """
    1. 提取目标层特征
    2. register 目标层梯度
    """

    def __init__(self, model, target_layers):
        # def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        # Define which layers you are going to extract
        # self.model_features = nn.Sequential(*list(self.model.children())[:4])
        # self.model_features = nn.Sequential(self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4,
        #                                     self.model.conv5)
        self.model_features = self.model.feature
        self.target_layers = target_layers
        self.gradients = list()

    def forward(self, x):
        return self.model_features(x)

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        target_activations = list()
        self.gradients = list()
        for name, module in self.model_features._modules.items():  # 遍历的方式遍历网络的每一层
            x = module(x)  # input 会经过遍历的每一层
            if name in self.target_layers:  # 设个条件，如果到了你指定的层， 则继续
                x.register_hook(self.save_gradient)  # 利用hook来记录目标层的梯度
                target_activations += [x]  # 这里只取得目标层的features
        x = x.view(x.size(0), -1)  # reshape成 全连接进入分类器
        x = self.model.dense(x)  # 进入分类器
        return target_activations, x,


def preprocess_image(img):
    """
    预处理层
    将图像进行标准化处理
    """
    mean = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()[:, :, ::-1]  # BGR > RGB

    # 标准化处理， 将bgr三层都处理
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - mean[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))  # transpose HWC > CHW
    preprocessed_img = t.from_numpy(preprocessed_img)  # totensor
    preprocessed_img.unsqueeze_(0)
    input = t.tensor(preprocessed_img, requires_grad=True)

    return input


def show_cam_on_image(img, mask, name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)  # 利用色彩空间转换将heatmap凸显
    heatmap = np.float32(heatmap) / 255  # 归一化
    cam = heatmap + np.float32(img)  # 将heatmap 叠加到原图
    cam = cam / np.max(cam)
    cv2.imwrite(savedir + name, np.uint8(255 * cam))  # 生成图像

    # cam = cam[:, :, ::-1]  # BGR > RGB
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.uint8(255 * cam))


class GradCam():
    """
    GradCam主要执行
    1.提取特征（调用FeatureExtractor)
    2.反向传播求目标层梯度
    3.实现目标层的CAM图
    """

    # def __init__(self, model, target_layer_names):
    def __init__(self, target_layer_names):
        self.model = net
        self.extractor = FeatureExtractor(self.model, target_layer_names)
        # self.extractor = FeatureExtractor(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):
        features, output = self.extractor(input)  # 这里的feature 对应的就是目标层的输出， output是图像经过分类网络的输出
        output.data
        one_hot = output.max()  # 取1000个类中最大的值

        # nn.Sequential(self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4,
        #               self.model.conv5)
        # self.model.features.zero_grad()  # 梯度清零
        # self.model.classifier.zero_grad()  # 梯度清零

        self.model.feature.zero_grad()
        self.model.dense.zero_grad()
        one_hot.backward(retain_graph=True)  # 反向传播之后，为了取得目标层梯度

        grad_val = self.extractor.get_gradients()[-1].data.numpy()
        # 调用函数get_gradients(),  得到目标层求得的梯

        target = features[-1]
        # features 目前是list 要把里面relu层的输出取出来, 也就是我们要的目标层 shape(1, 512, 14, 14)
        target = target.data.numpy()[0, :]  # (1, 512, 14, 14) > (512, 14, 14)

        # weights = np.mean(grad_val, axis=(2, 3))[0, :]  # array shape (512, ) 求出relu梯度的 512层 每层权重
        weights = np.mean(grad_val, axis=(0, 1))
        cam = np.zeros(target.shape[1:])  # 做一个空白map，待会将值填上
        # (14, 14)  shape(512, 14, 14)tuple  索引[1:] 也就是从14开始开始

        # for loop的方式将平均后的权重乘上目标层的每个feature map， 并且加到刚刚生成的空白map上
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
            # w * target[i, :, :]
            # target[i, :, :] = array:shape(14, 14)
            # w = 512个的权重均值 shape(512, )
            # 每个均值分别乘上target的feature map
            # 在放到空白的14*14上（cam)
            # 最终 14*14的空白map 会被填满

        cam = cv2.resize(cam, (224, 224))  # 将14*14的featuremap 放大回224*224
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam



# grad_cam = GradCam(model=net, target_layer_names=["7"])
# for i in range(13):
#     grad_cam = GradCam(target_layer_names=["%d" % i])
#     print(i)
#
#     img = cv2.imread('./dataset/test/up/Omega-02-Jun-2019-2-53176.jpg')  # 读取图像
#     img = np.float32(cv2.resize(img, (227, 227))) / 255  # 为了丢到vgg16要求的224*224 先进行缩放并且归一化
#     input = preprocess_image(img)
#     mask = grad_cam(input)
#     show_cam_on_image(img, mask, i, "up")
#
#     img = cv2.imread('./dataset/test/stay/Patamon-10-Jun-2019-1-123255.jpg')  # 读取图像
#     img = np.float32(cv2.resize(img, (227, 227))) / 255  # 为了丢到vgg16要求的224*224 先进行缩放并且归一化
#     input = preprocess_image(img)
#     mask = grad_cam(input)
#     show_cam_on_image(img, mask, i, "stay")
#
#     img = cv2.imread('./dataset/test/right/Patamon-10-Jun-2019-1-43386.jpg')  # 读取图像
#     img = np.float32(cv2.resize(img, (227, 227))) / 255  # 为了丢到vgg16要求的224*224 先进行缩放并且归一化
#     input = preprocess_image(img)
#     mask = grad_cam(input)
#     show_cam_on_image(img, mask, i, "right")
#
#     img = cv2.imread('./dataset/test/down/Patamon-10-Jun-2019-1-140615.jpg')  # 读取图像
#     img = np.float32(cv2.resize(img, (227, 227))) / 255  # 为了丢到vgg16要求的224*224 先进行缩放并且归一化
#     input = preprocess_image(img)
#     mask = grad_cam(input)
#     show_cam_on_image(img, mask, i, "down")

# 指明被遍历的文件夹
rootdir = r'../feature_eyes/'
hp_pic_list = os.listdir(rootdir)
grad_cam = GradCam(target_layer_names=["10"])
for htmp in hp_pic_list:
    print(htmp[0:23])
    if htmp[0:23] == '12-1-Omega-25-Jun-2019_':
        currentPath = os.path.join(rootdir, htmp)
        # currentPath = r'./1-1-Omega-01-Jun-2019.mp4_1.jpg'
        print('the fulll name of the file is :' + currentPath)
        img = cv2.imread(currentPath)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)
        mask = grad_cam(input)
        show_cam_on_image(img, mask, htmp)
