import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from PIL import Image
from torch import nn
import torch as t
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,models
import torch.optim as optim
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

        # 网络前向传播过程
        # RuntimeError: size mismatch, m1: [1000 x 6400], m2: [9216 x 4096]
        # All you have to care is b = c and you are done: m1: [a x b], m2: [c x d]
        # m1 is [a x b] which is [batch size x in features] in features不是输入图像大小，输入图像为96*96时为256，输入图像为227*227时为9216
        # m2 is [c x d] which is [ in features x out features]
        self.dense = t.nn.Sequential(
            t.nn.Linear(6400, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(0.5),
            t.nn.Linear(4096, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(0.5),
            t.nn.Linear(4096, 50)
        )

    def forward(self, x):
        feature_out = self.feature(x)
        res = feature_out.view(feature_out.size(0), -1)
        print("res" + str(res.shape))
        out = self.dense(res)
        return out

# model=CNN()

model = t.load('../models/sgd_1024_0.001.pkl')
print(model)
# print(model)
# print(model.features)
# model.load_state_dict(torch.load('../models/sgd_4_0.001.pkl'))
# print(model.feature)

def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):

    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)

    # img = img.unsqueeze(0)
    # print(transform)
    # 获取模型输出的feature/score
    model.eval()
    # feature_2=model.classifier(img)
    # print(img.shape) #
    feature_1=model.feature(img)
    print('feature1',feature_1[0].shape)
    res = feature_1.view(feature_1.size(0), -1)
    output = model.dense(res)
    print('out',output)

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    print(pred)
    pred_class = output[0][pred]
    # pred_class = output[:, pred]
    print(pred_class)

    feature_1.register_hook(extract)
    # pred_class.backward()  # 计算梯度
    pred_class.backward()
    grads = features_grad  # 获取梯度
    print('grads',grads)
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    print('pooled',pooled_grads.shape)
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    feature_1 = feature_1[0]
    feature_1 = feature_1.permute(1, 2, 0) # 变成（6，6，256）

    # print('pooled[1]',pooled_grads[1],pooled_grads[2])
    for i in range(256):
        # features[i, ...] *= pooled_grads[i, ...]
        feature_1[:,:,i] *= pooled_grads[i]

    heatmap = feature_1.detach().numpy()
    print('0',heatmap.shape)  #（6，6，256）


    heatmap = np.mean(heatmap, axis=-1)
    print('1,',heatmap.shape)  # （6，6）
    heatmap = np.maximum(heatmap, 0)
    print('2',heatmap.shape)
    heatmap /= np.max(heatmap)
    # print('3',heatmap) # （6，6）

    print('heatmap',heatmap.shape)


    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    # cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
    cv2.imwrite('saved_image_1024_0.001.jpg', superimposed_img)
Mytransform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                                transforms.ToTensor()])
draw_CAM(model,'../../data/dataset/test/left/Omega-02-Jul-2019-1-10066.jpg','./',Mytransform)
# draw_CAM(model,'Omega-02-Jul-2019-1-10014.jpg','./',Mytransform)
