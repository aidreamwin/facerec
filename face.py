# -*- coding: utf-8 -*-

import numpy as np 
import pickle
import cv2
import dlib
import io
import sys,os,glob,operator

# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()

# 2.加载人脸关键点检测器
predictor_path = "models/shape_predictor_5_face_landmarks.dat"
# predictor_path = "models/shape_predictor_68_face_landmarks.dat"
sp = dlib.shape_predictor(predictor_path)

# 3. 加载人脸识别模型
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

#人眼识别器分类器
classfier=cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

class Facedec(object):
    """docstring for Facedec"""
    def __init__(self,face_path):
        super(Facedec, self).__init__()
        self.descriptors = []
        self.candidate = ["kk/unknow.jpg"]
        self.face_path = face_path
        self.load_know_person(False)

    def load_know_person(self,reload=False):
        descriptors_path = self.face_path+"/descriptors.pk"
        candidate_path = self.face_path+"/candidate.pk"
        if not reload and os.path.exists(descriptors_path) and os.path.exists(candidate_path):
            with open(descriptors_path, 'rb') as f:
                self.descriptors = pickle.load(f)
            with open(candidate_path, 'rb') as f:
                self.candidate = pickle.load(f)
            return

        for f_name in glob.glob(os.path.join(self.face_path, "*.jpg")):
            print("Processing file: {}".format(f_name))
            img = cv2.imread(f_name)
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
                self.candidate.append(f_name)
        with open(descriptors_path, 'wb') as f:
            pickle.dump(self.descriptors, f)
        with open(candidate_path, 'wb') as f:
            pickle.dump(self.candidate, f)

    def check(self,file_name="S01E03.mkv"):
        #cv2.namedWindow(file_name)
        #调用摄像头
        cap=cv2.VideoCapture(file_name)
        # cap.set(3,640) #设置分辨率
        # cap.set(4,480)

        fps = cap.get(cv2.CAP_PROP_FPS)
        v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        size = (v_w,v_h)
        print(size,fps)
        # out = cv2.VideoWriter('oto_other2.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

        total = 0
        while cap.isOpened():
            read,frame=cap.read()
            
            if not read:
                break
            total += 1
            # if total < 60*60*2:
            #     continue
            if total % 2 == 0:
                continue
            if total % 3 ==0:
                continue
            # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            #灰度转换
            grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #人脸检测
            Rects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (80, 80))
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
                    dist = [0.93]
                    for i in self.descriptors:
                        # dist_ = np.linalg.norm(i-v)
                        dist_=np.dot(i,v)/(np.linalg.norm(i)*(np.linalg.norm(v)))
                        # print(dist_)
                        dist.append(dist_)

                    c_d = dict(zip(self.candidate,dist))
                    # cd_sorted = sorted(c_d.iteritems(), key=lambda x:x[1])
                    cd_sorted = sorted(c_d.items(), key=operator.itemgetter(1), reverse=True)
                    # print(cd_sorted)
                    
                    # print(c_d)
                    name = cd_sorted[0][0].split("/")[-1][:-4]
                    name = name.split("_")[0]
                    # print("\n The person is: ",name)


                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (X + 6, H + 16), font, 0.7, (0, 0, 255), 1)
                    rate = round(cd_sorted[0][1],6)
                    cv2.putText(frame, "PR: {}".format(rate), (x+3, y-5), font, 0.5, (0, 0, 255), 1)
                    print("The person is: {},PR: {}".format(name,rate))
                    
                    
            cv2.imshow(file_name,frame)
            # cv2.waitKey(1000/int(30)) #延迟
            if cv2.waitKey(5)&0xFF==ord('q'):
                break
            # 保存视频
            # out.write(frame)
            # if total > 60*60:
            #     break
        #释放相关资源
        cap.release()
        # out.release()
        cv2.destroyAllWindows()


f = Facedec("know_pic/qy")
f.check("qy2_2.mkv")

# f.check("./oto_other.mp4")