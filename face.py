# -*- coding: utf-8 -*-

import numpy as np 
import cv2
 
#fourcc = cv2.VideoWriter_fourcc("D", "I", "B", " ")
#out = cv2.VideoWriter('frame_mosic.MP4',fourcc, 20.0, (640,480))
 
cv2.namedWindow("CaptureFace")
#调用摄像头
cap=cv2.VideoCapture("harry.mkv")
#人眼识别器分类器
classfier=cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

while cap.isOpened():
    read,frame=cap.read()
    if not read:
        break
    #灰度转换
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #人脸检测
    Rects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
    if len(Rects) > 0:
        for (x, y, w, h) in Rects:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			try:
				X = int(x*1)
				W = min(int((x + w)*1.0),frame.shape[1])
				Y = int(y*1)
				H = min(int((y + h)*1.0),frame.shape[0])

				face_image = cv2.resize(frame[Y:H, X:W], (W-X,H-Y))
				cv2.imshow("CaptureFace2",face_image)
			except Exception as e:
				print(x,y,w,h,face_image.shape)
			
    cv2.imshow("CaptureFace",frame)
    # cv2.waitKey(1000/int(30)) #延迟
    if cv2.waitKey(5)&0xFF==ord('q'):
        break
    # 保存视频
    #out.write(frame)
#释放相关资源
cap.release()
#out.release()
cv2.destroyAllWindows()