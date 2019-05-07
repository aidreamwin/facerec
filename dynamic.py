# -*- coding: utf-8 -*-
import configparser
import dlib
import cv2
class Dynamic(object):
	"""docstring for Dynamic"""
	def __init__(self,classfier_type):
		cf = configparser.ConfigParser()
		cf.read("config/config.cfg")
		predictor_path = cf.get("face_dlib", "predictor_path")
		face_rec_model_path = cf.get("face_dlib", "face_rec_model_path")
		classfier_path = cf.get("face_opencv", "classfier_path")
		# 1.加载正脸检测器
		self.detector = dlib.get_frontal_face_detector()
		# 2.加载人脸关键点检测器
		self.sp = dlib.shape_predictor(predictor_path)
		# 3. 加载人脸识别模型mmod_human_face_detector.dat
		self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
		#人眼识别器分类器
		self.classfier=cv2.CascadeClassifier(classfier_path)

		classfier_type = cf.get("face_classfier", classfier_type) # Thrones
		self.video_src = cf.get(classfier_type, "video_src")
		self.video_tmp = cf.get(classfier_type, "video_tmp")
		self.video_des = cf.get(classfier_type, "video_des")
		self.picture_src = cf.get(classfier_type, "picture_src")
		self.threshold = cf.getfloat(classfier_type, "threshold")
		self.mark_unknown = cf.getboolean(classfier_type, "mark_unknown")
		self.save_scale = cf.getfloat(classfier_type, "save_scale")
		self.reload_pic = cf.getboolean(classfier_type, "reload_pic")
		self.video_type = cf.get(classfier_type, "video_type")
		self.scaleFactor = cf.getfloat(classfier_type, "scaleFactor")
		self.minNeighbors = cf.getint(classfier_type, "minNeighbors")
		self.minSize = cf.getint(classfier_type, "minSize")
		self.begin_time = cf.getint(classfier_type, "begin_time")
		self.end_time = cf.getint(classfier_type, "end_time")
		

		