import cv2
import torch
import os 
import numpy as np
import const_parameters as cp
from preprocess import PreProcess as pp
from model import Classfier
from model import Classfier_r3d
from model import Classfier_r3d_b
from dataset import VideoDataSet
from torch.utils.data import DataLoader
main_path = "../opencv/test_videos"

# file = train_video_list[0]
train_video_list = [ "../opencv/test_videos/11.mp4" , "../opencv/test_videos/1.mp4"
                    , "../opencv/test_videos/5.mp4", "../opencv/test_videos/8.mp4"
                    , "../opencv/test_videos/9.mp4"
                    , "../opencv/test_videos/6.mp4", "../opencv/test_videos/10.mp4"
                    , "../opencv/test_videos/7.mp4" ]
# train_video_list , _ = pp.read_files(main_path,main_path)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
num_frames = 60
target_list = np.zeros(60)
catch_scale = 1
# 載入model
# dict = torch.load(os.path.join(cp.MODEL_PATH , f"{num_frames} frames.pth"))
dict = torch.load(os.path.join(cp.MODEL_PATH ,"60 frames train 0.86 test 0.91 (1.0 for binary).pth"))
model = Classfier_r3d()
model.load_state_dict(dict)


tensor_list = []
for file in train_video_list:
    cap = cv2.VideoCapture(file)
    print(file)
    # cap = cv2.VideoCapture(file[0])
    # print(file[0])
    if not cap.isOpened():
        print("Cannot open camera")
        continue   
    frames_received = 0
    face_list = []
    while frames_received < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        frames_received += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 將鏡頭影像轉換成灰階
        faces = face_cascade.detectMultiScale(gray)          # 偵測人臉
        
        # print(f"faces = {faces}")
        max = -1
        max_id = -1
        id = 0
        tem_list = []
        for (x, y, w, h) in faces:
            if w * h > max:
                max = w * h
                max_id += 1
                y1 = int (y + h * (1 - catch_scale) / 2)
                y2 = int (y + h - h * (1 - catch_scale) / 2)
                x1 = int (x + w * (1 - catch_scale) / 2)
                x2 = int (x + w - w * (1 - catch_scale) / 2)
                face = frame[y1:y2 , x1:x2] # y軸先才x軸
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)   # 標記人臉
                # print(f"{frames_received} : ({x},{y},{x+w},{y+h})")
                face = cv2.resize(face,[171,128]) # 縮小尺寸
                tem_list.append(face)
            id += 1
        if len(tem_list) == 0:
            face_list = []
            frames_received = 0
            continue
        # print(f"temlist len = {len(tem_list)} max_id = {max_id}")
        face = tem_list[max_id]
        cv2.imshow("face" , face)
        cv2.waitKey(1)
        face = np.transpose(face , (2,0,1)) # (H,W,C) -> (C,H,W)
        face = torch.tensor(face , dtype = torch.float32)
        face_list.append(face)

    if frames_received != num_frames:
        continue

    # 把list整合成一個4維tensor，並放入最終的tensor list
    video_tensor = torch.stack(face_list , dim = 0)
    video_tensor = torch.transpose(video_tensor,1,0) # (T,C,H,W) -> (C,T,H,W)
    tensor_list.append(video_tensor)
# torch.save(tensor_list , "tensor_list.pth")
cv2.destroyAllWindows()

# tensor_list = torch.load("../tensor_list.pth")
face_set = VideoDataSet(tensor_list , target_list)
face_loader = DataLoader(face_set)
i = 0
acc = 0
binary_acc = 0
for batch in face_loader:
    with torch.no_grad():
        tensor , _ = batch
        output = model(tensor)
        print(output)
        # 找出最值最大的標籤
        _ , pred = torch.max(output , dim = 1)
        result = pred.item()
        # target = pp.targetType(train_video_list[i])
        # print(f"Target is {target}")
        print(f"Result is {result}")
        # #if target == result:
        #     acc += 1
        #     binary_acc += 1
        # elif target == 1 and result == 2:
        #     binary_acc += 1
        # elif target == 2 and result == 1:
        #     binary_acc += 1
        
        i += 1
# print(f"Accuracy is {acc/len(face_loader):3.5f}")
# print(f"Binary Accuracy is {binary_acc/len(face_loader):3.5f}")
# print(f"# of frames : {num_frame}")

