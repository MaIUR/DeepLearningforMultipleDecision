# coding=utf-8
from PIL import Image
import os.path

# 指明被遍历的文件夹
rootdir = r'../label_eyes/'
label_pic_list = os.listdir(rootdir)
feature_dir = r'../feature_eyes/'

# total	         1615540
# train	         969324      +++
# validation     323108      +
# test	         323108      +
# count取余5，若为0、1、2则存入train，若为3则存入validation，若为4则存入test，使得大致比例为3：1：1
count = 0

for label_pic in label_pic_list:
    currentPath = os.path.join(rootdir, label_pic)
    print('the fulll name of the file is :' + currentPath)
    img = Image.open(currentPath)
    # featurePath = os.path.join(feature_dir, label_pic)
    # feature = Image.open(featurePath)
    # print(img)
    width = img.width
    height = img.height
    print(img.format, img.size, img.mode)
    image_list = []
    for x in range(height):
        for y in range(width):
            pixel = img.getpixel((y, x))
            image_list.append(pixel)
    black = 0
    total = len(image_list)
    print('总像素个数')
    print(total)
    for i in range(total):
        if (image_list[i] == (0, 0, 0)):
            black += 1
    print('黑色像素个数')
    print(black)
    print('白色像素个数')
    white = total - black
    print(white)

    # 分类：无操作-stay，左-left，上-up，右-right，下-down
    # 如果白色像素个数为0，patamon的数据，删掉
    if white == 0:
        if (os.path.exists(currentPath)):
            os.remove(currentPath)
            print(currentPath + 'delete success')
        else:
            print("要删除的文件不存在！")
    # 如果白色像素个数为7530，类别为stay
    elif white > 5000:
        temp = count % 5
        if temp == 0 or temp == 1 or temp == 2:
            # feature.save(r"../../../../data/marui/dataset_2/train/stay" + '/' + label_pic)  # 存储到指定的分类下
            # print("save success! " + label_pic)
            count += 1
            f1 = open('trainset.txt', 'r+')
            f1.read()
            f1.write(feature_dir + label_pic + " 0\n")
            f1.close()
        elif temp == 3:
            # feature.save(r"./dataset/validation/stay" + '/' + label_pic)  # 存储到指定的分类下
            # print("save success! " + label_pic)
            count += 1
            f2 = open('validationset.txt', 'r+')
            f2.read()
            f2.write(feature_dir + label_pic + " 0\n")
            f2.close()
        elif temp == 4:
            # feature.save(r"./dataset/test/stay" + '/' + label_pic)  # 存储到指定的分类下
            # print("save success! " + label_pic)
            count += 1
            f3 = open('testset.txt', 'r+')
            f3.read()
            f3.write(feature_dir + label_pic + " 0\n")
            f3.close()
    else:
        # left
        box_left = (0, 50, 90, 120)  # 设置左、上、右、下的像素
        img_left = img.crop(box_left)  # 图像裁剪
        left_width = img_left.width
        left_height = img_left.height
        img_left_list = []
        for x in range(left_height):
            for y in range(left_width):
                pixel = img_left.getpixel((y, x))
                img_left_list.append(pixel)
        left_white = 0
        left_total = len(img_left_list)
        for i in range(left_total):
            if (img_left_list[i] == (255, 255, 255)):
                left_white += 1
        print('left白色像素个数' + str(left_white))

        # up
        box_up = (50, 0, 120, 90)  # 设置左、上、右、下的像素
        img_up = img.crop(box_up)  # 图像裁剪
        up_width = img_up.width
        up_height = img_up.height
        img_up_list = []
        for x in range(up_height):
            for y in range(up_width):
                pixel = img_up.getpixel((y, x))
                img_up_list.append(pixel)
        up_white = 0
        up_total = len(img_up_list)
        for i in range(up_total):
            if (img_up_list[i] == (255, 255, 255)):
                up_white += 1
        print('up白色像素个数' + str(up_white))

        # right
        box_right = (80, 50, 170, 120)  # 设置左、上、右、下的像素
        img_right = img.crop(box_right)  # 图像裁剪
        right_width = img_right.width
        right_height = img_right.height
        img_right_list = []
        for x in range(right_height):
            for y in range(right_width):
                pixel = img_right.getpixel((y, x))
                img_right_list.append(pixel)
        right_white = 0
        right_total = len(img_right_list)
        for i in range(right_total):
            if (img_right_list[i] == (255, 255, 255)):
                right_white += 1
        print('right白色像素个数' + str(right_white))

        # down
        box_down = (50, 80, 120, 170)  # 设置左、上、右、下的像素
        img_down = img.crop(box_down)  # 图像裁剪
        down_width = img_down.width
        down_height = img_down.height
        img_down_list = []
        for x in range(down_height):
            for y in range(down_width):
                pixel = img_down.getpixel((y, x))
                img_down_list.append(pixel)
        down_white = 0
        down_total = len(img_down_list)
        for i in range(down_total):
            if (img_down_list[i] == (255, 255, 255)):
                down_white += 1
        print('down白色像素个数' + str(down_white))

        # 判断哪一部分白色像素点最多
        m = max(left_white, up_white, right_white, down_white)
        if left_white == m:
            temp = count % 5
            if temp == 0 or temp == 1 or temp == 2:
                # feature.save(r"./dataset/train/left" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f1 = open('trainset.txt', 'r+')
                f1.read()
                f1.write(feature_dir + label_pic + " 1\n")
                f1.close()
            elif temp == 3:
                # feature.save(r"./dataset/validation/left" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f2 = open('validationset.txt', 'r+')
                f2.read()
                f2.write(feature_dir + label_pic + " 1\n")
                f2.close()
            elif temp == 4:
                # feature.save(r"./dataset/test/left" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f3 = open('testset.txt', 'r+')
                f3.read()
                f3.write(feature_dir + label_pic + " 1\n")
                f3.close()
        elif up_white == m:
            temp = count % 5
            if temp == 0 or temp == 1 or temp == 2:
                # feature.save(r"./dataset/train/up" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f1 = open('trainset.txt', 'r+')
                f1.read()
                f1.write(feature_dir + label_pic + " 2\n")
                f1.close()
            elif temp == 3:
                # feature.save(r"./dataset/validation/up" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f2 = open('validationset.txt', 'r+')
                f2.read()
                f2.write(feature_dir + label_pic + " 2\n")
                f2.close()
            elif temp == 4:
                # feature.save(r"./dataset/test/up" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f3 = open('testset.txt', 'r+')
                f3.read()
                f3.write(feature_dir + label_pic + " 2\n")
                f3.close()
        elif right_white == m:
            temp = count % 5
            if temp == 0 or temp == 1 or temp == 2:
                # feature.save(r"./dataset/train/right" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f1 = open('trainset.txt', 'r+')
                f1.read()
                f1.write(feature_dir + label_pic + " 3\n")
                f1.close()
            elif temp == 3:
                # feature.save(r"./dataset/validation/right" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f2 = open('validationset.txt', 'r+')
                f2.read()
                f2.write(feature_dir + label_pic + " 3\n")
                f2.close()
            elif temp == 4:
                # feature.save(r"./dataset/test/right" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f3 = open('testset.txt', 'r+')
                f3.read()
                f3.write(feature_dir + label_pic + " 3\n")
                f3.close()
        elif down_white == m:
            temp = count % 5
            if temp == 0 or temp == 1 or temp == 2:
                # feature.save(r"./dataset/train/down" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f1 = open('trainset.txt', 'r+')
                f1.read()
                f1.write(feature_dir + label_pic + " 4\n")
                f1.close()
            elif temp == 3:
                # feature.save(r"./dataset/validation/down" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f2 = open('validationset.txt', 'r+')
                f2.read()
                f2.write(feature_dir + label_pic + " 4\n")
                f2.close()
            elif temp == 4:
                # feature.save(r"./dataset/test/down" + '/' + label_pic)  # 存储到指定的分类下
                # print("save success! " + label_pic)
                count += 1
                f3 = open('testset.txt', 'r+')
                f3.read()
                f3.write(feature_dir + label_pic + " 4\n")
                f3.close()
        else:
            f4 = open('../DefineLabel_error.txt', 'r+')
            f4.read()
            f4.write(label_pic + "\n")
            f4.close()
print(count)
