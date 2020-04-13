# coding=utf-8
from PIL import Image
import os.path

# 指明被遍历的文件夹
rootdir = r'../raw_eyes'
for img_name in os.listdir(rootdir):
    currentPath = os.path.join(rootdir, img_name)
    print('the fulll name of the file is :' + currentPath)
    try:
        img = Image.open(currentPath)
        # print(img)
        print(img.format, img.size, img.mode)
        # img.show()
        box1 = (0, 65, 740, 870)  # 设置左、上、右、下的像素
        image1 = img.crop(box1)  # 图像裁剪
        box2 = (750, 350, 920, 520)
        image2 = img.crop(box2)
        image1.save(r"../feature_eyes/" + img_name)  # 存储裁剪得到的图像
        image2.save(r"../label_eyes/" + img_name)
    except(OSError, NameError):
        f = open('../CutImages_error.txt', 'r+')
        f.read()
        f.write("\n%s\n" % img_name)
        f.close()

# img = Image.open('F:\\study\\机器学习\\experiment\\srtp\\2.0\\raw_eyes\\1-1-Omega-01-Jun-2019.mp4_1.jpg')
# # print(img)
# print(img.format, img.size, img.mode)
# # img.show()
# box1 = (0, 65, 740, 870)  # 设置左、上、右、下的像素
# image1 = img.crop(box1)  # 图像裁剪
# box2 = (750, 350, 920, 520)
# image2 = img.crop(box2)
# image1.save(r"../feature_eyes" + '/1-1-Omega-01-Jun-2019.mp4_1.jpg')  # 存储裁剪得到的图像
# image2.save(r"../label_eyes" + '/1-1-Omega-01-Jun-2019.mp4_1.jpg')
