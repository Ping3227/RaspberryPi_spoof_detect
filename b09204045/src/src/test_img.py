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
# 載入model
model = md.ImgClassfier_self()
model.load_state_dict(dict.DICT)
print(model._get_name())

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


def test(train_video_list , target_list):
# 計數器
    i = -1
    acc = 0
    true_positive = 0 # 把真的認成真的
    false_positive = 0 # 把假的認成真的
    true_negative = 0 # 把假的認成假的
    false_negative = 0 # 把真的認成假的
    test_size = 0
    fp_list = []
    fn_list = []

    # 每一個影片取n禎照片檢視結果
    for file in train_video_list:

        i += 1
        face_list = [] # 要丟進model的臉們
        mark_list = []
        real = 0       # 認為是真臉的數量
        fake = 0       # 認為是假臉的數量
        save_list = []

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

            mark_list.append([frame , tem_list[max_id]])

            # 把最大的臉整理成一個可以讓模型讀的tensor
            face = tem_list[max_id][0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)   # 標記人臉
            if show == True:
                cv2.imshow("face" , frame)
                cv2.waitKey(1)

            save_list.append(face)

            face = np.transpose(face , (2,0,1)) # (H,W,C) -> (C,H,W)
            face = torch.tensor(face , dtype = torch.float32)
            face_list.append(face)

        j = 0
        # 把這n張臉丟進去模型統計答案
        face_set = ImgDataSet_Test(face_list)
        face_loader = DataLoader(face_set)
        for batch in face_loader:
            with torch.no_grad():
                tensor = batch
                output = model(tensor)
                _ , pred = torch.max(output , dim = 1) # 找出最值最大的標籤
                result = pred.item()
                if result == 0:
                    real += 1
                    if save: cv2.imwrite(os.path.join(save_path , f"real-{i}-{j}.jpg"), save_list[j])
                else:
                    fake += 1
                    if save: cv2.imwrite(os.path.join(save_path , f"fake-{i}-{j}.jpg"), save_list[j])
            j += 1

        # 找出target
        if mode == "test":
            target = pp.targetType(file)
            if target == 2:
                target = 1
            target_list.append(target)

        target = target_list[i]

        # 找出答案
        if real > fake:
            final_result = 0
        else:
            final_result = 1

        if final_result == target:
            acc += 1

        # 找出標記點
        face_to_mark = mark_list[len(mark_list) - 1][0]
        x1 = mark_list[len(mark_list) - 1][1][1]
        y1 = mark_list[len(mark_list) - 1][1][2]
        x2 = mark_list[len(mark_list) - 1][1][3]
        y2 = mark_list[len(mark_list) - 1][1][4]

        print(f"Real : {real} Fake : {fake}")
        print(f"Target is {target}")
        print(f"Result is {final_result}")

        # 算f score順便標出結果在影片
        if final_result == 0:
            cv2.rectangle(face_to_mark, (x1,y1), (x2,y2), (0, 255, 0), 2)   # 標記人臉
            if target == 0: # 把真的認成真的
                true_positive += 1
            else: # 把假的認成真的
                false_positive += 1
                if mode == "self":
                    fp_list.append(video_name.split('/')[-1])
                else:
                    fp_list.append(video_name.split('/')[-1].split("\\")[-1])
        else:
            cv2.rectangle(face_to_mark, (x1,y1), (x2,y2), (0, 0, 255), 2)   # 標記人臉
            if target == 0: # 把真的認成假的
                false_negative += 1
                if mode == "self":
                    fn_list.append(video_name.split('/')[-1])
                else:
                    fn_list.append(video_name.split('/')[-1].split("\\")[-1])
            else: # 把假的認成假的
                true_negative += 1

        if show == True:
            cv2.imshow("face" , frame)
            cv2.waitKey(10)

        test_size += 1

    cv2.destroyAllWindows()

    return test_size , acc , true_positive , true_negative , false_positive , false_negative , fp_list , fn_list

def toRate(test_size , acc , true_positive , true_negative , false_positive , false_negative):
    accuracy = acc/test_size
    true_positive_rate = true_positive/test_size
    true_negative_rate = true_negative/test_size
    false_positive_rate = false_positive/test_size
    false_negative_rate = false_negative/test_size
    return test_size , accuracy , true_positive_rate , true_negative_rate , false_positive_rate , false_negative_rate

if mode == "self":
    self_1m_test_size , self_1m_accuracy , self_1m_tp , self_1m_tn , self_1m_fp , self_1m_fn , self_1m_fp_list , self_1m_fn_list = test(self_1m_video_list , self_1m_target_list)
    self_test_size , self_accuracy , self_tp , self_tn , self_fp , self_fn, self_fp_list , self_fn_list = test(self_video_list , self_target_list)
    test_size = self_test_size + self_1m_test_size
    accuracy = self_accuracy + self_1m_accuracy
    tp = self_tp + self_1m_tp
    tn = self_tn + self_1m_tn
    fp = self_fp + self_1m_fp
    fn = self_fn + self_1m_fn
    fp_list = self_fp_list + self_1m_fp_list
    fn_list = self_fn_list + self_1m_fn_list
    self_test_size , self_accuracy , self_tp , self_tn , self_fp , self_fn = toRate(self_test_size , self_accuracy , self_tp , self_tn , self_fp , self_fn)
    self_1m_test_size , self_1m_accuracy , self_1m_tp , self_1m_tn , self_1m_fp , self_1m_fn = toRate(self_1m_test_size , self_1m_accuracy , self_1m_tp , self_1m_tn , self_1m_fp , self_1m_fn)
else:
    test_size , accuracy , tp , tn , fp ,fn , fp_list ,fn_list = test(test_video_list , test_target_list)

test_size , accuracy , tp , tn , fp ,fn = toRate(test_size , accuracy , tp , tn , fp ,fn)
print()
print(f"Model:{model._get_name()}")
print("Dataset #  Acc     TP      TN      FP      FN")
if mode == "self":
    print(f"Normal  {self_test_size:2} {self_accuracy:3.5f} {self_tp:3.5f} {self_tn:3.5f} {self_fp:3.5f} {self_fn:3.5f}")
    print(f"1m      {self_1m_test_size:2} {self_1m_accuracy:3.5f} {self_1m_tp:3.5f} {self_1m_tn:3.5f} {self_1m_fp:3.5f} {self_1m_fn:3.5f}")
print(f"Total   {test_size:2} {accuracy:3.5f} {tp:3.5f} {tn:3.5f} {fp:3.5f} {fn:3.5f}")
print(f"False Positive : {fp_list}")
print(f"False Negative : {fn_list}")
