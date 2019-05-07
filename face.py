# -*- coding: utf-8 -*-

import numpy as np 
import pickle
import cv2
import dlib
import io,sys,os,glob,operator,time
from dynamic import Dynamic

class Facedec(object):
    """docstring for Facedec"""
    def __init__(self,dynamic):
        super(Facedec, self).__init__()
        self.descriptors = []
        self.candidate = ["kk/unknown.jpg"]
        self.dynamic = dynamic
        self.load_know_person()

    def load_know_person(self):
        reload = self.dynamic.reload_pic
        descriptors_path = os.path.join(self.dynamic.picture_src, "descriptors.pk")
        candidate_path = os.path.join(self.dynamic.picture_src, "candidate.pk")
        if not reload and os.path.exists(descriptors_path) and os.path.exists(candidate_path):
            with open(descriptors_path, 'rb') as f:
                self.descriptors = pickle.load(f)
            with open(candidate_path, 'rb') as f:
                self.candidate = pickle.load(f)
            return
        for f_name in glob.glob(os.path.join(self.dynamic.picture_src, "*.jpg")):
            print("Processing file: {}".format(f_name))
            img = cv2.imread(f_name)
            dets = self.dynamic.detector(img, 1)
            for k, d in enumerate(dets):
                # 2.关键点检测
                shape = self.dynamic.sp(img, d)
                # 3.描述子提取，128D向量
                face_descriptor = self.dynamic.facerec.compute_face_descriptor(img, shape)
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
        if self.dynamic.end_time != 0 and self.dynamic.begin_time > self.dynamic.end_time:
            print("ERROR,begin_time > end_time")
            return
        self.orivideo_name = file_name

        sum_time = self.get_time(file_name)
        #调用摄像头
        cap=cv2.VideoCapture(file_name)
        help(cap)

        fps = cap.get(cv2.CAP_PROP_FPS)
        v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out_mp4 = os.path.join(self.dynamic.video_tmp, file_name.split("/")[-1])
        out = cv2.VideoWriter(self.out_mp4, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (v_w,v_h))

        total = 0
        while cap.isOpened():
            read,frame=cap.read()
            if not read:
                break
            total += 1

            if self.dynamic.end_time != 0 and total < self.dynamic.begin_time*fps:
                continue
            if self.dynamic.end_time != 0 and total > self.dynamic.end_time*fps:
                break

            if total % (fps*10) < 1:
                run_rate = round(total / (fps * sum_time) * 100.00,3)
                s = "{}%,{}/{}".format(run_rate,int(total/fps),int(sum_time))
                print(s)
            #灰度转换
            grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #人脸检测
            Rects = self.dynamic.classfier.detectMultiScale(grey, 
                scaleFactor = self.dynamic.scaleFactor, 
                minNeighbors = self.dynamic.minNeighbors, 
                minSize = (self.dynamic.minSize, self.dynamic.minSize))
            if len(Rects) > 0:
                for (x, y, w, h) in Rects:
                    X,W,Y,H = self.get_pos(frame,x,y,w,h)
                    face_image = cv2.resize(frame[Y:H, X:W], (W-X,H-Y))
                    v = self.compute_face_vector(face_image)
                    cd_sorted = self.compute_cosine(v)
                    name = self.get_person_name(cd_sorted)
                    pr = round(cd_sorted[0][1],6)
                    self.draw_pic(frame,x,y,w,h,name,pr)
                    # if name != "unknown":
                    #     print("The person is: {},PR: {}".format(name,pr))
                    
                    
            cv2.imshow(file_name,frame)
            if cv2.waitKey(5)&0xFF==ord('q'):
                break
            # 保存视频
            save_scale = self.dynamic.save_scale
            frame = cv2.resize(frame, (0, 0), fx=save_scale, fy=save_scale)
            out.write(frame)
            # if total > fps*10:
            #     break
        #释放相关资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        self.get_mp3()
        self.merge_video()

    def s2str(self,s):
        h = int(s / 3600)
        tmp = s % 3600
        m = int(tmp / 60)
        tmp = tmp % 60
        s = tmp
        if h < 10:
            h = "0{}".format(h)
        else:
            h = "{}".format(h)
        if m < 10:
            m = "0{}".format(m)
        else:
            m = "{}".format(m)
        if s < 10:
            s = "0{}".format(s)
        else:
            s = "{}".format(s)

        str_s = "{}:{}:{}".format(h,m,s)
        return str_s

    def get_mp3(self):
        # ffmpeg -ss 0:0:00 -t 0:0:20 -i qy2_2.mkv 20.mp3
        begin_time = self.s2str(self.dynamic.begin_time)
        end_time = self.dynamic.end_time - self.dynamic.begin_time
        name = self.orivideo_name.split("/")[-1] + ".mp3"
        self.mp3 = os.path.join(self.dynamic.video_tmp, name)
        if end_time <= 0:
            cmd = "ffmpeg -i {} {}".format(self.orivideo_name,self.mp3)
        else:
            cmd = "ffmpeg -ss {} -t {} -i {} {}".format(begin_time,end_time,self.orivideo_name,self.mp3)
        r = os.system(cmd)
        if r != 0:
            print("ERROR [{}]".format(cmd))

    def get_time(self,file_name):
        cmd = "ffmpeg -i {} 2>&1 | grep 'Duration' | cut -d ' ' -f 4 | sed s/,//".format(file_name)
        result = os.popen(cmd).read()
        print(result)
        h = float(result.split(":")[0])
        m = float(result.split(":")[1])
        s = float(result.split(":")[2])
        sum_time = h*60*60 + m*60 + s
        if sum_time <= 0:
            return 1
        return sum_time


    def merge_video(self):
        name = self.orivideo_name.split("/")[-1]
        out_name = os.path.join(self.dynamic.video_des, name)
        cmd = "ffmpeg -i {} -i {} -c copy {}".format(self.out_mp4,self.mp3,out_name)
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
        shape = self.dynamic.sp(face_image, d)
        face_descriptor = self.dynamic.facerec.compute_face_descriptor(face_image, shape)
        return np.array(face_descriptor)

    def draw_pic(self,frame,x,y,w,h,name,pr):
        if pr <= self.threshold:
            return
        X,W,Y,H = self.get_pos(frame,x,y,w,h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (X + 6, H + 16), font, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, "PR: {}".format(pr), (x+3, y-5), font, 0.5, (0, 0, 255), 1)
    def run(self):
        for f_name in glob.glob(os.path.join(self.dynamic.video_src, self.dynamic.video_type)):
            f.check(f_name)

if __name__ == '__main__':

    d = Dynamic("Thrones")
    f = Facedec(d)
    f.check("video/Thrones/02/S02E07.mkv")