# encoding:utf-8
import cv2
import os.path

rootdir = r'../vedio_eyes'
vlist = os.listdir(rootdir)
for video in vlist:
    videodir = os.path.join(rootdir, video)
    vc = cv2.VideoCapture(videodir)
    c = 1
    # 获取视频fps
    fps = vc.get(cv2.CAP_PROP_FPS)
    # 获取视频总帧数
    frame_all = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    print("[INFO] 视频FPS: {}".format(fps))
    print("[INFO] 视频总帧数: {}".format(frame_all))
    print("[INFO] 视频时长: {}s".format(frame_all / fps))
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
        # VideoToImage_error
        f = open('VideoToImage_error.txt', 'r+')
        f.read()
        f.write("\n%s\n" % video)
        f.close()
    while rval and c<frame_all-30:
        rval, frame = vc.read()
        cv2.imwrite('../raw_eyes/' + video + '_' + str(c) + '.jpg', frame)
        c = c + 1
        # cv2.waitKey(1)
    vc.release()
