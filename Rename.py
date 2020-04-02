# encoding:utf-8
import os.path

# 1153192张图片
rootdir = r'../../../../data/marui/raw_eyes/'
# rootdir = r'../test/'
list = os.listdir(rootdir)
count = 0
for image in list:
    os.rename(os.path.join(rootdir, image), os.path.join(rootdir, image[8:len(image)]))
    count += 1
print(count)