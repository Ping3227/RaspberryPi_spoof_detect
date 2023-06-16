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
import RPi.GPIO as GPIO
import time as time_fun
main_path = "../opencv/test_videos"
mode = "camera" # test or self
show = False
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
num_frames  = 7
catch_scale = 1
max_time    = 5  # 最大容忍抓不到的禎數

# 載入model
dict = torch.load(os.path.join("../ImgClassfier_self 124994 0.947 batch 256 lr 0.0001.pth"),map_location = torch.device('cpu'))
model = ImgClassfier_self()
model.load_state_dict(dict)

# 計數器
i = 0
acc = 0
test_size = 0
final_result =0


if mode == "camera":

    cap = Picamera2()
    camera_config = cap.create_preview_configuration(main={"size": (1280,704)}) ## , lores={"size": (640, 480)}, display="lores" (1920,1088) 1280 704 640 480
    cap.configure(camera_config)
    
    cap.start()      

# 每一個影片取n禎照片檢視結果

while(1):

    # 亮黃燈不閃

    frame = cap.capture_array("main")
    frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)       # 將鏡頭影像轉換成灰階
    faces = face_cascade.detectMultiScale(gray)          # 偵測人臉

    if len(faces) == 0:
        continue

    # 亮閃黃燈    

    face_list = [] # 要丟進model的臉們
    mark_list = [] 
    real = 0       # 認為是真臉的數量
    fake = 0       # 認為是假臉的數量
   
    #if not cap.isOpened():
        #print("Cannot open camera")
        #exit()  

    # 開始讀禎
    frames_received = 0
    time = 0
    while frames_received < num_frames and time < max_time:
        time += 1
        print(f"time = {time}")

        # 如果已經到底了 跳出迴圈
        # ret, frame = cap.read()
        frame = cap.capture_array("main")

        #if not ret:
            #print("Cannot receive frame")
            #break

        # cv2.imwrite(f"/home/capteam005/face/{time}.jpg" , frame)
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


    if show == True:
        cv2.imshow("face" , frame)
        cv2.waitKey(500)

    i += 1
    test_size += 1

    cv2.destroyAllWindows()

    # 使用 BCM 編號
        
    GPIO.setmode(GPIO.BCM)
    if(final_result == 1): ##false  red light 
        # 操作 GPIO 4（Pin 7）
        pin = 27
    else : ## true green light 
        pin = 17
    # 設定為 GPIO 為輸入模式
    GPIO.setup(pin, GPIO.OUT)
    # 設定 GPIO 輸出值為高電位
    GPIO.output(pin, GPIO.HIGH)
        
    # 等待一秒鐘 lighting for 1 second 
    time_fun.sleep(2)
    # 設定 GPIO 輸出值為低電位
    GPIO.output(pin, GPIO.LOW)
    GPIO.cleanup()
    print("light ")