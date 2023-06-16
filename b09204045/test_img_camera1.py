import cv2
import torch
import os 
import numpy as np
# import const_parameters as cp
from preprocess import PreProcess as pp
from model import ImgClassfier_self
from dataset import ImgDataSet_Test
from torch.utils.data import DataLoader
from picamera2 import Picamera2
main_path = "../opencv/test_videos"
mode = "camera" # test or self
show = True
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
num_frames = 7
catch_scale = 1

# 載入model
dict = torch.load(os.path.join("../ImgClassfier_self 124994 0.947 batch 256 lr 0.0001.pth"),map_location = torch.device('cpu'))
model = ImgClassfier_self()
model.load_state_dict(dict)

# 計數器
i = 0
acc = 0
true_positive = 0 # 把真的認成真的
false_positive = 0 # 把假的認成真的
true_negative = 0 # 把假的認成假的
false_negative = 0 # 把真的認成假的
test_size = 0

# 每一個影片取n禎照片檢視結果
for x in range(1):

    face_list = [] # 要丟進model的臉們
    mark_list = [] 
    real = 0       # 認為是真臉的數量
    fake = 0       # 認為是假臉的數量

    # 讀取影片檔案
    if mode == "camera":
        
        cap = Picamera2()
        camera_config = cap.create_preview_configuration(main={"size": (1920,1088)}) ## , lores={"size": (640, 480)}, display="lores"
        cap.configure(camera_config)
        
        cap.start()           
        
    # else:
    #     cap = cv2.VideoCapture(file[0])
    #     print(file[0])
    #if not cap.isOpened():
        #print("Cannot open camera")
        #exit()  

    # 開始讀禎
    frames_received = 0
    time = 0
    while frames_received < num_frames and time < 50:
        time += 1
        print(f"time = {time}")

        # 如果已經到底了 跳出迴圈
        ##ret, frame = cap.read()
        frame = cap.capture_array("main")

        #if not ret:
            #print("Cannot receive frame")
            #break

        cv2.imwrite(f"/home/capteam005/face/{time}.jpg" , frame)
        frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        if show == True:
            cv2.imshow("face" , frame)
            cv2.waitKey(1000)
            #picam2.start_preview(Preview.QT)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)       # 將鏡頭影像轉換成灰階
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
        print(f"抓到{frames_received}張臉")
        mark_list.append([frame , tem_list[max_id]])

        # 把最大的臉整理成一個可以讓模型讀的tensor        
        face = tem_list[max_id][0]

        cv2.imwrite(f"face/{frames_received}.jpg" , face)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)   # 標記人臉
        if show == True:
            cv2.imshow("face" , frame)
            cv2.waitKey(100)
       
        face = np.transpose(face , (2,0,1)) # (H,W,C) -> (C,H,W)
        face = torch.tensor(face , dtype = torch.float32)
        face_list.append(face)

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
            else:
                fake += 1

    # 找出答案
    if real > fake:
        final_result = 0
    else:
        final_result = 1

    
    # 找出標記點
    face_to_mark = mark_list[len(mark_list) - 1][0]
    x1 = mark_list[len(mark_list) - 1][1][1]
    y1 = mark_list[len(mark_list) - 1][1][2]
    x2 = mark_list[len(mark_list) - 1][1][3]
    y2 = mark_list[len(mark_list) - 1][1][4]

    print(f"Real : {real} Fake : {fake}")
    print(f"Result is {final_result}")

    # # 算f score順便標出結果在影片
    # if final_result == 0: 
    #     cv2.rectangle(face_to_mark, (x1,y1), (x2,y2), (0, 255, 0), 2)   # 標記人臉
    #     if target == 0: # 把真的認成真的
    #         true_positive += 1
    #     else: # 把假的認成真的
    #         false_positive += 1
    # else: 
    #     cv2.rectangle(face_to_mark, (x1,y1), (x2,y2), (0, 0, 255), 2)   # 標記人臉
    #     if target == 0: # 把真的認成假的
    #         false_negative += 1
    #     else: # 把假的認成假的
    #         true_negative += 1

    if show == True:
        cv2.imshow("face" , frame)
        cv2.waitKey(500)

    i += 1
    test_size += 1

cv2.destroyAllWindows()

# # 印出準確率
# true_positive_rate = true_positive/test_size
# true_negative_rate = true_negative/test_size
# false_positive_rate = false_positive/test_size
# false_negative_rate = false_negative/test_size
# print(test_size)
# print(f"Accuracy is {acc/test_size:3.5f}")
# print(f"True Positive: {true_positive_rate:3.5f} False Positive: {false_positive_rate:3.5f}")
# print(f"True Negative: {true_negative_rate:3.5f} False Negative: {false_negative_rate:3.5f}")