# coding=utf-8
import cv2

img_root = r'../hotmap/sgd_4_0.001/12-1-Omega-25-Jun-2019_'  # 这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
fps = 30  # 保存视频的FPS，可以适当调整
size = (224, 224)
# 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
videoWriter = cv2.VideoWriter('../hotmap/sgd_4_0.001.avi', fourcc, fps, size)  # 最后一个是保存图片的尺寸

# for(i=1;i<471;++i)
for i in range(1, 3136):
    print(i)
    frame = cv2.imread(img_root + str(i) + '.jpg')
    videoWriter.write(frame)
videoWriter.release()
