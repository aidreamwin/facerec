# -*- coding: utf-8 -*-

import os
import time

def get_mp3(orivideo_name):
    # ffmpeg -ss 0:0:00 -t 0:0:20 -i qy2_2.mkv 20.mp3
    n_time = int(time.time())
    mp3 = "{}_{}.mp3".format(orivideo_name,n_time)
    cmd = "ffmpeg -i {} {}".format(orivideo_name,mp3)
    # cmd = "ffmpeg -ss 0:0:00 -t 0:0:10 -i {} {}".format(self.orivideo_name,self.mp3)
    r = os.system(cmd)
    if r != 0:
        print("ERROR [{}]".format(cmd))
    return mp3

# mp3 = get_mp3("video/qy2_1.mp4")

import pygame  
pygame.mixer.init()  
mp3 = "video/qy2_1.mp4_1557053209.mp3"
print("播放音乐1")  
track = pygame.mixer.music.load(mp3)  
# pygame.mixer.music.play()

# time.sleep(10)#播放10秒
# pygame.mixer.music.stop()#停止播放

# os.system()

while True:
	#检查音乐流播放，有返回True，没有返回False
	#如果没有音乐流则选择播放
	if pygame.mixer.music.get_busy()==False:
		pygame.mixer.music.play()