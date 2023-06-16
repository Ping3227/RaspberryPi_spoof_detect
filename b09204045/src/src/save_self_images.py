import cv2
import torch
import os
import numpy as np
import const_parameters as cp
import model as md
import test_dicts as dict
from preprocess import PreProcess as pp
from dataset import ImgDataSet_Test
from torch.utils.data import DataLoader
main_path = "../opencv/test_videos"
save_path = "../opencv/test_videos/results"
mode = "self" # test or self
show = False
save = True
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
num_frames = 33
catch_scale = 1

if mode == "self":
    self_video_list = [ # 真人
                         "../opencv/test_videos/20.mp4" , "../opencv/test_videos/21.mp4" ,
                         "../opencv/test_videos/7.mp4"  , "../opencv/test_videos/1.mp4"  ,
                         "../opencv/test_videos/9.mp4"  ,  "../opencv/test_videos/8.mp4" ,
                         "../opencv/test_videos/12.mp4" , 

                        # # 假人
                         "../opencv/test_videos/2.mp4"  , "../opencv/test_videos/55.mp4" ,
                         "../opencv/test_videos/3.mp4"  , "../opencv/test_videos/54.mp4" ,
                         "../opencv/test_videos/4.mp4"  , "../opencv/test_videos/11.mp4" ,
                         "../opencv/test_videos/61.mp4" , "../opencv/test_videos/62.mp4" ,
                         "../opencv/test_videos/51.mp4" , "../opencv/test_videos/52.mp4" ,
                         "../opencv/test_videos/53.mp4" , 
                       ]
    
    self_1m_video_list = [ # 真人_1公尺
                            "../opencv/test_videos/48.mp4" , "../opencv/test_videos/49.mp4"  ,
                            "../opencv/test_videos/46.mp4" , "../opencv/test_videos/47.mp4"  ,
                            "../opencv/test_videos/44.mp4" , "../opencv/test_videos/45.mp4"  ,
                            "../opencv/test_videos/42.mp4" , "../opencv/test_videos/43.mp4"  ,
                            "../opencv/test_videos/23.mp4" , "../opencv/test_videos/22.mp4"  ,
                            "../opencv/test_videos/37.mp4" , "../opencv/test_videos/39.mp4"  ,
                            "../opencv/test_videos/26.mp4" , "../opencv/test_videos/14.mp4"  ,
                           # 假人_1公尺
                            "../opencv/test_videos/56.mp4" , "../opencv/test_videos/59.mp4" ,
                            "../opencv/test_videos/57.mp4" , "../opencv/test_videos/60.mp4" ,
                            "../opencv/test_videos/58.mp4" , "../opencv/test_videos/29.mp4" ,
                            "../opencv/test_videos/50.mp4" , "../opencv/test_videos/28.mp4" ,
                            "../opencv/test_videos/36.mp4" , "../opencv/test_videos/41.mp4" ,
                            "../opencv/test_videos/34.mp4" , "../opencv/test_videos/33.mp4" ,
                            "../opencv/test_videos/30.mp4" , "../opencv/test_videos/31.mp4" ,
                            "../opencv/test_videos/32.mp4" , 
                         ]
    self_size     = len(self_video_list)
    self_1m_size  = len(self_1m_video_list)
    num_self_fake = 11
    num_self_1m_fake = 15

    self_target_list = np.zeros(self_size,dtype= int)
    for i in range(self_size - num_self_fake,self_size):
        self_target_list[i] = 1
    print(f"Normal : {self_target_list}")

    self_1m_target_list = np.zeros(self_1m_size,dtype= int)
    for i in range(self_1m_size - num_self_1m_fake,self_1m_size):
        self_1m_target_list[i] = 1
    print(f"1m : {self_1m_target_list}")
else:
    test_video_list , _ = pp.read_files(main_path,main_path)
    test_target_list = []

idx = -1
def save(train_video_list , target_list , idx):
    i = -1
    # 每一個影片取n禎照片檢視結果
    for file in train_video_list:
        idx += 1
        i += 1
        save_list = []
        target = target_list[i]

        # 讀取影片檔案
        video_name = file if mode == "self" else file[0]
        cap = cv2.VideoCapture(video_name)
        print(video_name)
        
        if not cap.isOpened():
            print("Cannot open camera")
            continue

        # 開始讀禎
        frames_received = 0
        while frames_received < num_frames:

            # 如果已經到底了 跳出迴圈
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                break

            # cv2.imwrite(f"images/{frames_received}.jpg" , frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       # 將鏡頭影像轉換成灰階
            faces = face_cascade.detectMultiScale(gray)          # 偵測人臉

            max = -1      # 最大的臉面積
            max_id = -1   # 最大的臉的ID
            id = 0        # 這張臉的ID
            tem_list = [] # 儲存同一張照片裡全部的臉

            # 只抓出最大的臉
            for (x, y, w, h) in faces:
                if w * h > max:
                    max = w * h
                    max_id += 1
                    y1 = int (y + h * (1 - catch_scale) / 2)
                    y2 = int (y + h - h * (1 - catch_scale) / 2)
                    x1 = int (x + w * (1 - catch_scale) / 2)
                    x2 = int (x + w - w * (1 - catch_scale) / 2)
                    face = frame[y1:y2 , x1:x2] # y軸先才x軸
                    face = cv2.resize(face,[128,128]) # 縮小尺寸
                    tem_list.append([face,x1,y1,x2,y2])
                id += 1

            # 如果這禎沒有臉 跳過
            if len(tem_list) == 0:
                continue
            frames_received += 1
            face = tem_list[max_id][0]

            if show == True:
                cv2.imshow("face" , frame)
                cv2.waitKey(1)
            if target == 0:
                cv2.imwrite(os.path.join(cp.SELF_REAL_PATH , f"real-{idx}-{frames_received}.jpg"), face)
            else:
                cv2.imwrite(os.path.join(cp.SELF_FAKE_PATH , f"fake-{idx}-{frames_received}.jpg"), face)
    cv2.destroyAllWindows()
    return i

idx = save(self_video_list , self_target_list , idx)
idx = save(self_1m_video_list , self_1m_target_list , idx)
