# -*- coding: utf-8 -*-

import numpy as np 
import cv2
import dlib
import io
import sys,os,glob,operator

# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()

# 2.加载人脸关键点检测器
# predictor_path = "models/shape_predictor_5_face_landmarks.dat"
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
sp = dlib.shape_predictor(predictor_path)

# 3. 加载人脸识别模型
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

#人眼识别器分类器
classfier=cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

class Facedec(object):
    """docstring for Facedec"""
    def __init__(self):
        super(Facedec, self).__init__()
        self.descriptors = []
        self.candidate = ["kk/unknow.jpg"]
        self.load_know_person()

    def load_know_person(self):
        for f in glob.glob(os.path.join("know_pic", "*.jpg")):
            print("Processing file: {}".format(f))
            img = cv2.imread(f)
            dets = detector(img, 1)
            for k, d in enumerate(dets):
                # d = dlib.rectangle(0,0,img.shape[0],img.shape[1])
                # 2.关键点检测
                shape = sp(img, d)
                # 3.描述子提取，128D向量
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                # 转换为numpy array
                v = np.array(face_descriptor)
                self.descriptors.append(v)
                self.candidate.append(f)

    def check(self,file_name="S01E03.mkv"):
        #fourcc = cv2.VideoWriter_fourcc("D", "I", "B", " ")
        #out = cv2.VideoWriter('frame_mosic.MP4',fourcc, 20.0, (640,480))
        cv2.namedWindow(file_name)
        #调用摄像头
        cap=cv2.VideoCapture(file_name)

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
                    name = "uknow"
                    #try:
                    X = int(x*1)
                    W = min(int((x + w)*1.0),frame.shape[1])
                    Y = int(y*1)
                    H = min(int((y + h)*1.0),frame.shape[0])

                    face_image = cv2.resize(frame[Y:H, X:W], (W-X,H-Y))
                    d = dlib.rectangle(0,0,face_image.shape[0],face_image.shape[1])
                    shape = sp(face_image, d)
                    face_descriptor = facerec.compute_face_descriptor(face_image, shape)
                    v = np.array(face_descriptor)
                    dist = [0.9]
                    for i in self.descriptors:
                        dist_ = np.linalg.norm(i-v)
                        print(dist_)
                        dist.append(dist_)

                    c_d = dict(zip(self.candidate,dist))
                    #cd_sorted = sorted(c_d.iteritems(), key=lambda x:x[1])
                    cd_sorted = sorted(c_d.items(), key=operator.itemgetter(1), reverse=False)
                    # print(cd_sorted)
                    
                    # print(cd_sorted[0][0].split("/"))
                    name = cd_sorted[0][0].split("/")[-1][:-4]
                    print("\n The person is: ",name)


                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (X + 6, H + 16), font, 0.7, (0, 0, 255), 1)
                    
                    
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


f = Facedec()
f.check()