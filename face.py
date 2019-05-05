# -*- coding: utf-8 -*-

import numpy as np 
import pickle
import cv2
import dlib 
import io,sys,os,glob,operator,time

# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()

# 2.加载人脸关键点检测器
predictor_path = "models/shape_predictor_5_face_landmarks.dat"
# predictor_path = "models/shape_predictor_68_face_landmarks.dat"
sp = dlib.shape_predictor(predictor_path)

# 3. 加载人脸识别模型mmod_human_face_detector.dat
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

#人眼识别器分类器
classfier=cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

class Facedec(object):
    """docstring for Facedec"""
    def __init__(self,face_path):
        super(Facedec, self).__init__()
        self.descriptors = []
        self.candidate = ["kk/unknown.jpg"]
        self.face_path = face_path
        self.load_know_person(True)

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
        if not os.path.exists(file_name):
            print("ERROR,{} not exist".format(file_name))
            return
        self.orivideo_name = file_name
        #cv2.namedWindow(file_name)
        #调用摄像头
        cap=cv2.VideoCapture(file_name)
        # cap.set(3,640) #设置分辨率
        # cap.set(4,480)

        fps = cap.get(cv2.CAP_PROP_FPS)
        v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out_mp4 = file_name+'.mp4'
        out = cv2.VideoWriter(self.out_mp4, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (v_w,v_h))

        total = 0
        while cap.isOpened():
            read,frame=cap.read()
            if not read:
                break
            total += 1
            #灰度转换
            grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #人脸检测
            Rects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (80, 80))
            if len(Rects) > 0:
                for (x, y, w, h) in Rects:
                    X,W,Y,H = self.get_pos(frame,x,y,w,h)
                    face_image = cv2.resize(frame[Y:H, X:W], (W-X,H-Y))
                    # face_image = cv2.resize(face_image, (0, 0), fx=0.5, fy=0.5)
                    v = self.compute_face_vector(face_image)
                    cd_sorted = self.compute_cosine(v)
                    name = self.get_person_name(cd_sorted)
                    pr = round(cd_sorted[0][1],6)
                    self.draw_pic(frame,x,y,w,h,name,pr)
                    print("The person is: {},PR: {}".format(name,pr))
                    
                    
            cv2.imshow(file_name,frame)
            if cv2.waitKey(5)&0xFF==ord('q'):
                break
            # 保存视频
            out.write(frame)
            # if total > fps*10:
            #     break
        #释放相关资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        self.get_mp3()
        self.merge_video()

    def get_mp3(self):
        # ffmpeg -ss 0:0:00 -t 0:0:20 -i qy2_2.mkv 20.mp3
        n_time = int(time.time())
        self.mp3 = "{}_{}.mp3".format(self.orivideo_name,n_time)
        cmd = "ffmpeg -i {} {}".format(self.orivideo_name,self.mp3)
        # cmd = "ffmpeg -ss 0:0:00 -t 0:0:10 -i {} {}".format(self.orivideo_name,self.mp3)
        r = os.system(cmd)
        if r != 0:
            print("ERROR [{}]".format(cmd))


    def merge_video(self):
        n_time = int(time.time())
        cmd = "ffmpeg -i {} -i {} -c copy {}.out_{}.mkv".format(self.out_mp4,self.mp3,self.orivideo_name,n_time)
        r = os.system(cmd)
        if r != 0:
            print("ERROR [{}]".format(cmd))
        else:
            cmd = "rm -rf {} {}".format(self.out_mp4,self.mp3)
            os.system(cmd)

    def compute_cosine(self,v,threshold=0.93):
        self.threshold = threshold
        dist = [threshold]
        for i in self.descriptors:
            dist_=np.dot(i,v)/(np.linalg.norm(i)*(np.linalg.norm(v)))
            dist.append(dist_)
        c_d = dict(zip(self.candidate,dist))
        cd_sorted = sorted(c_d.items(), key=operator.itemgetter(1), reverse=True)
        return cd_sorted

    def compute_euclidean(self,v,threshold=0.4):
        self.threshold = threshold
        dist = [threshold]
        for i in self.descriptors:
            dist_ = np.linalg.norm(i-v)
            dist.append(dist_)
        c_d = dict(zip(self.candidate,dist))
        cd_sorted = sorted(c_d.items(), key=operator.itemgetter(1), reverse=False)
        return cd_sorted

    def get_person_name(self,cd_sorted):
        name = cd_sorted[0][0].split("/")[-1][:-4]
        name = name.split("_")[0]
        return name

    def get_pos(self,frame,x,y,w,h):
        X = int(x*1)
        W = min(int((x + w)*1.0),frame.shape[1])
        Y = int(y*1)
        H = min(int((y + h)*1.0),frame.shape[0])
        return X,W,Y,H

    def compute_face_vector(self,face_image):
        d = dlib.rectangle(0,0,face_image.shape[0],face_image.shape[1])
        shape = sp(face_image, d)
        face_descriptor = facerec.compute_face_descriptor(face_image, shape)
        return np.array(face_descriptor)

    def draw_pic(self,frame,x,y,w,h,name,pr):
        if pr <= self.threshold:
            return
        X,W,Y,H = self.get_pos(frame,x,y,w,h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (X + 6, H + 16), font, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, "PR: {}".format(pr), (x+3, y-5), font, 0.5, (0, 0, 255), 1)


if __name__ == '__main__':
    f = Facedec("know_pic/qy")
    f.check("video/s02/S02E07.mkv")