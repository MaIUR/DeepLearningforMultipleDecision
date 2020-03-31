import cv2
vc=cv2.VideoCapture("/home/hqd/PycharmProjects/1/1/19.MOV")
c=1
if vc.isOpened():
    rval,frame=vc.read()
else:
    rval=False
while rval:
    rval,frame=vc.read()
    cv2.imwrite('/home/hqd/PycharmProjects/1/1/19/'+str(c)+'.jpg',frame)
    c=c+1
    cv2.waitKey(1)
vc.release()