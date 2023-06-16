import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import psutil

# 有問題的檔案
defect = ["Test/spoof/020/020-2-3-3-2.face" , "Test/spoof/021/021-2-3-2-1.face" , "Test/spoof/061/061-2-3-2-1.face",
          "Test/spoof/061/061-2-3-3-1.face" , "Test/spoof/141/141-2-3-2-1.face" , "Test/spoof/141/141-2-3-3-1.face",
          "Test/spoof/161/161-2-3-2-1.face" , "Test/spoof/161/161-2-3-3-1.face" , "Train/spoof/041/041-2-3-2-1.face",
          "Train/spoof/041/041-2-3-3-1.face", "Train/spoof/081/081-2-3-2-1.face" , "Train/spoof/101/101-2-3-2-1.face",
          "Train/spoof/101/101-2-3-3-1.face", "Train/spoof/121/121-2-3-2-1.face", "Train/spoof/121/121-2-3-3-1.face"]
class PreProcess():

    # 讀取SiW檔名
    def readSiW(main_path):
        # _w代表是利用walk讀進來的檔名
        file_list_w = []
        video_list_w = []
        face_list_w = []
        for root, dir, files in os.walk(main_path):
            for f in files:
                if(f.endswith(".mov") or f.endswith(".face")):
                    file_list_w.append(os.path.join(root , f))
        file_list = sorted(file_list_w)
        
        # 因為SiW有些face檔沒有對應到影片檔，如果有對到才放入檔案清單
        for i in range(len(file_list)):
            if i < len(file_list) - 1:

                # 檢查是否在瑕疵清單，如果有就跳過
                if len(defect) > 0:
                    found = False
                    for defect_file in defect:
                        if file_list[i] == os.path.join(main_path , defect_file):
                            defect.remove(defect_file)
                            # print(f"Found a defect file {file_list[i]}")
                            # print(f"# of defect files remained : {len(defect)}")
                            found = True 
                            break
                    if found == True:continue

                if file_list[i].endswith(".face") and file_list[i + 1].endswith(".mov"):
                    face_name = file_list[i].split('.')[-2]
                    video_name = file_list[i + 1].split('.')[-2]
                    if(face_name == video_name):
                        video_list_w.append([file_list[i+1] , "SiW"])
                        face_list_w.append([file_list[i] , "SiW"])
        
        # 把檔案排好以利於裁切
        video_list = sorted(video_list_w)
        face_list = sorted(face_list_w)
        return video_list , face_list
    
    # 讀取OULU檔名
    def readOULU(main_path):
        video_list_w = []
        face_list_w = []
        for root, dir, files in os.walk(main_path):
            for f in files:
                if(f.endswith(".avi")):
                    video_list_w.append([os.path.join(root , f) , "OULU"])
                elif(f.endswith(".txt")):
                    face_list_w.append([os.path.join(root , f) , "OULU"])
        video_list = sorted(video_list_w)
        face_list = sorted(face_list_w)
        return video_list , face_list
    
    # 讀取指定路徑的全部檔名，回傳影片檔和臉部檔list(每一個元素為[檔名,資料集名稱]) e.g.[video.mov , 'SiW']
    def read_files(SiW_Path , OULU_Path):
        o_video , o_face = PreProcess.readOULU(OULU_Path)
        s_video , s_face = PreProcess.readSiW(SiW_Path)
        video_list = s_video + o_video   
        face_list = s_face + o_face
        return video_list , face_list
    
    # 讀取SiW檔名，回傳[root,file_name,dataset]
    def readSiWImg(main_path):
        # _w代表是利用walk讀進來的檔名
        file_list_w = []
        video_list_w = []
        face_list_w = []
        for root, dir, files in os.walk(main_path):
            for f in files:
                if(f.endswith(".mov") or f.endswith(".face")):
                    file_list_w.append([root , f])
                    # print([root , f])
        file_list = sorted(file_list_w)
        
        # 因為SiW有些face檔沒有對應到影片檔，如果有對到才放入檔案清單
        for i in range(len(file_list)):
            if i < len(file_list) - 1:

                # 檢查是否在瑕疵清單，如果有就跳過
                if len(defect) > 0:
                    found = False
                    for defect_file in defect:
                        if file_list[i] == defect_file:
                            defect.remove(defect_file)
                            # print(f"Found a defect file {file_list[i]}")
                            # print(f"# of defect files remained : {len(defect)}")
                            found = True 
                            break
                    if found == True:continue

                if file_list[i][1].endswith(".face") and file_list[i + 1][1].endswith(".mov"):
                    face_name = file_list[i][1].split('.')[-2]
                    video_name = file_list[i + 1][1].split('.')[-2]
                    if(face_name == video_name):
                        video_list_w.append([file_list[i+1][0] ,file_list[i+1][1] , "SiW"])
                        face_list_w.append([file_list[i][0] ,file_list[i][1] , "SiW"])
        
        # 把檔案排好以利於裁切
        video_list = sorted(video_list_w)
        face_list = sorted(face_list_w)
        return video_list , face_list
    
    # 讀取OULU檔名
    def readOULUImg(main_path):
        video_list_w = []
        face_list_w = []
        for root, dir, files in os.walk(main_path):
            for f in files:
                if(f.endswith(".avi")):
                    video_list_w.append([root ,f , "OULU"])
                elif(f.endswith(".txt")):
                    face_list_w.append([root ,f , "OULU"])
        video_list = sorted(video_list_w)
        face_list = sorted(face_list_w)
        return video_list , face_list

    # 讀取指定路徑的全部檔名，回傳[root,file_name,dataset]
    def read_filesImg(SiW_Path , OULU_Path):
        o_video , o_face = PreProcess.readOULUImg(OULU_Path)
        s_video , s_face = PreProcess.readSiWImg(SiW_Path)
        video_list = s_video + o_video   
        face_list = s_face + o_face
        return video_list , face_list
    
    # 把read_files回傳的影片檔案全部轉成4維Tensor list，每一個Tensor維度(Channels = 3 , 禎數 , 高度 , 寬度)
    def toTensors(video_list , face_list ,  # read_files 回傳的列表
                  num_frames ,              # 欲讀禎數
                  h ,w ,                    # 裁切大小
                  scale                     # OULU臉框的scale
                  ):
        tensor_list = []
        target_list = []
        strange_list = []
        for i in tqdm(range(len(video_list))):
            # 把影片讀進來
            cap = cv2.VideoCapture(video_list[i][0])

            # 如果影片讀不進來，跳過這個影片
            if cap.isOpened() != True:
                strange_list.append([video_list[i][0] , "Cannot read file"])
                continue
            sus , positions = PreProcess.facePositions(face_list[i])

            # 如果face檔有問題，跳過這個影片
            if sus == False:
                strange_list.append([video_list[i][0] , "Face file has defect"])
                continue
            
            frame_list = []
            cropped_frames = 0
            j = -1 # index of current frame
            while cropped_frames < num_frames:
                j = j + 1 # j從0開始

                # 抓取一禎影片
                ret = cap.grab()

                # 如果影片已經讀到底了，跳出迴圈            
                if not ret: 
                    break

                # 如果臉部位置全為0或是出現負數，跳過這禎，重頭開始算
                wrong_positions = False
                if any(positions[j]) == False:
                    wrong_positions = True

                for position in positions[j]:
                    if position < 0:
                        wrong_positions = True
                        break

                if wrong_positions:
                    frame_list = []
                    cropped_frames = 0
                    continue
                
                # 解析這一禎
                ret , frame = cap.retrieve()
                if not ret: 
                    continue
                
                # 裁切臉部轉成三維tensor並加入暫時的list
                frame = PreProcess.faceCrop(frame , j , face_list[i][1] , positions ,h , w , scale)
                frame = np.transpose(frame , (2,0,1)) # (H,W,C) -> (C,H,W)
                frame = torch.tensor(frame , dtype = torch.float32)
                frame_list.append(frame)
                cropped_frames = cropped_frames + 1

            # 如果跳出迴圈時禎數不對，跳過這個影片
            if len(frame_list) != num_frames:
                if len(frame_list) < num_frames:
                    strange_list.append([video_list[i][0] , "Valid frames are not enough"])
                elif len(frame_list) > num_frames:
                    strange_list.append([video_list[i][0] , "Valid frames are too much"])
                continue

            # 把list整合成一個4維tensor，並放入最終的tensor list
            video_tensor = torch.stack(frame_list , dim = 0)
            video_tensor = torch.transpose(video_tensor,1,0) # (T,C,H,W) -> (C,T,H,W)
            tensor_list.append(video_tensor)

            # 找到target加入target list
            target = PreProcess.targetType(video_list[i])
            target_list.append(target)
            # print(video_list[i] , target)

        return tensor_list , target_list , strange_list
    
    def toImageTensors(video_list , face_list ,  # read_files 回傳的列表
                  num_frames ,                   # 欲讀禎數
                  h ,w ,                         # 裁切大小
                  scale                          # OULU臉框的scale
                  ):
        tensor_list = []
        target_list = []
        strange_list = []
        for i in tqdm(range(len(video_list))):
            # 把影片讀進來
            cap = cv2.VideoCapture(video_list[i][0])

            # 如果影片讀不進來，跳過這個影片
            if cap.isOpened() != True:
                strange_list.append([video_list[i][0] , "Cannot read file"])
                continue
            sus , positions = PreProcess.facePositions(face_list[i])

            # 如果face檔有問題，跳過這個影片
            if sus == False:
                strange_list.append([video_list[i][0] , "Face file has defect"])
                continue
            
            cropped_frames = 0
            j = -1 # index of current frame
            while cropped_frames < num_frames:
                j = j + 1 # j從0開始

                # 抓取一禎影片
                ret = cap.grab()

                # 如果影片已經讀到底了，跳出迴圈            
                if not ret: 
                    break

                # 如果臉部位置全為0或是出現負數，跳過這禎
                wrong_positions = False
                if any(positions[j]) == False:
                    wrong_positions = True

                for position in positions[j]:
                    if position < 0:
                        wrong_positions = True
                        break

                if wrong_positions:
                    continue
                
                # 解析這一禎
                ret , frame = cap.retrieve()
                if not ret: 
                    continue
                
                # 裁切臉部轉成三維tensor並加入暫時的list
                frame = PreProcess.faceCrop(frame , j , face_list[i][1] , positions ,h , w , scale)
                frame = np.transpose(frame , (2,0,1)) # (H,W,C) -> (C,H,W)
                frame = torch.tensor(frame , dtype = torch.float32)
                tensor_list.append(frame)
                cropped_frames = cropped_frames + 1
                # 找到target加入target list
                target = PreProcess.targetType(video_list[i])
                target_list.append(target)

            # # 如果跳出迴圈時禎數不對，跳過這個影片
            # if len(frame_list) != num_frames:
            #     if len(frame_list) < num_frames:
            #         strange_list.append([video_list[i][0] , "Valid frames are not enough"])
            #     elif len(frame_list) > num_frames:
            #         strange_list.append([video_list[i][0] , "Valid frames are too much"])
            #     continue

            # # 把list整合成一個4維tensor，並放入最終的tensor list
            # video_tensor = torch.stack(frame_list , dim = 0)
            # video_tensor = torch.transpose(video_tensor,1,0) # (T,C,H,W) -> (C,T,H,W)
            # tensor_list.append(video_tensor)

            # # 找到target加入target list
            # target = PreProcess.targetType(video_list[i])
            # target_list.append(target)
            # print(video_list[i] , target)

        return tensor_list , target_list , strange_list
    
    # 把read_files回傳的影片檔案全部轉成照片儲存
    def toImages(save_path, 
                 video_list , face_list ,  # read_files 回傳的列表
                  num_frames ,                   # 欲讀禎數
                  h ,w ,                         # 裁切大小
                  scale,                         # OULU臉框的scale
                  index
                  ):
        strange_list = []

        
        for i in tqdm(range(len(video_list))):
            video_read_name = os.path.join(video_list[i][0] , video_list[i][1])
            face_read_name  = os.path.join(face_list[i][0] , face_list[i][1])
            save_name = video_list[i][1]
            dataset   = video_list[i][2]

            # 把影片讀進來
            cap = cv2.VideoCapture(video_read_name)

            # 如果影片讀不進來，跳過這個影片
            if cap.isOpened() != True:
                strange_list.append([video_read_name , "Cannot read file"])
                continue

            sus , positions = PreProcess.facePositions([face_read_name , dataset])

            # 如果face檔有問題，跳過這個影片
            if sus == False:
                strange_list.append([face_read_name , "Face file has defect"])
                continue
            
            cropped_frames = 0
            j = -1 # index of current frame

            while cropped_frames < num_frames:
                j = j + 1 # j從0開始

                # 抓取一禎影片
                ret = cap.grab()

                # 如果影片已經讀到底了，跳出迴圈            
                if not ret: 
                    break

                # 如果臉部位置全為0或是出現負數，跳過這禎
                wrong_positions = False
                if any(positions[j]) == False:
                    wrong_positions = True

                for position in positions[j]:
                    if position < 0:
                        wrong_positions = True
                        break

                if wrong_positions:
                    continue
                
                # 解析這一禎
                ret , frame = cap.retrieve()
                if not ret: 
                    continue
                
                # 裁切臉部並儲存
                frame = PreProcess.faceCrop(frame , j , dataset , positions ,h , w , scale)
                # print(f"{save_path}/{save_name.split('.')[-2]}-{cropped_frames}.jpg")
                cv2.imwrite(f"{save_path}/{save_name.split('.')[-2]}-{cropped_frames}.jpg" , frame)

                cropped_frames = cropped_frames + 1
                
            index += 1

        return  strange_list
    
    # 輸入[臉部檔名,資料集] 回傳 bool : 臉部位置是否有效 , 2d int array : 每一禎的檔案內容
    def facePositions(face):
        # 打開face檔並讀取臉部位置
        with open(face[0]) as f:
            lines = f.readlines()

        # 如果是空檔案，則為無效
        if any(lines) == False:
            return False , []
        
        # 提取檔案內容
        for i in range(len(lines)):
            if face[1] == "SiW":
                lines[i] = lines[i].replace('\n' , '').split(' ')[:4]
            else:
                lines[i] = lines[i].replace('\n' , '').split(',')[1:] 

        # 改成 int array
        positions = np.zeros((len(lines),4),dtype = int)
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                if lines[i][j] == '':
                    return False , []
                positions[i][j] = int(lines[i][j])

        return True , positions

    # 裁切臉部
    def faceCrop(frame , frame_index , dataset , positions , h ,w ,scale):
        if(dataset == "SiW"):
            
            # SiW face檔內的標註點形式：(x1 , y1 , x2 , y2) 原點在左上角
            pt1 = (positions[frame_index][0] , positions[frame_index][1])
            pt2 = (positions[frame_index][2], positions[frame_index][3])
        else: # OULU
            
            # 計算左右眼中點
            left = (positions[frame_index][0],positions[frame_index][1])
            right = (positions[frame_index][2],positions[frame_index][3])
            x_mid = (left[0] + right[0]) / 2
            y_mid = (left[1] + right[1]) / 2
            
            # 計算框框大小
            width = right[0] * scale
            height = right[1] * scale

            # 框出臉部
            x1 = x_mid - width / 2
            y1 = y_mid - height / 2
            x2 = x_mid + width / 2
            y2 = y_mid + height / 2
            pt1 = (int(x1) , int(y1))
            pt2 = (int(x2) , int(y2))
       
        # 裁切臉部後回傳
        frame_crop = frame[pt1[1]:pt2[1] , pt1[0]:pt2[0]]
        frame_crop = cv2.resize(frame_crop , [w ,h])
        # cv2.imshow("face" , frame_crop)
        # cv2.waitKey(5)
        return frame_crop
    
    # 檢查是否有影片和臉部檔名不對稱
    def num_mismatch(video_list , face_list):
        same = False
        false = 0
        for i in range(len(video_list)):
            same = False
            v = video_list[i][0].split('.')[-2]
            f = face_list[i][0].split('.')[-2]
            if(v == f):
                same = True
                # print(v + " == " + f)
            if same == False: false = false + 1
        return false
    
    # 檢查是否有影片和臉部檔名不對稱(for照片)
    def num_mismatchImg(video_list , face_list):
        # print("fuck")
        same = False
        false = 0
        for i in range(len(video_list)):
            #print(i)
            same = False
            v = video_list[i][1].split('.')[-2]
            f = face_list[i][1].split('.')[-2]
            if(v == f):
                same = True
                # print(v + " == " + f)
            if same == False: false = false + 1
        return false
        
    # 輸入[影片檔名,資料集] 回傳 Type 0 : Live / 1 : Image attack / 2 : Video attack
    def targetType(video):
        # SiW : SubjectID_SensorID_TypeID_MediumID_SessionID 
        # Type 1 : Live / Type 2 : Image attack / Type 3 : Video attack
        # OULU : Phone_Session_User_Type
        # Type 1 : Live / Type 2,3 : Image attack / Type 4,5 : Video attack

        if video[1] == 'SiW':
            type = int(video[0].split('-')[-3]) - 1
        else:
            type = int(video[0].split('_')[-1].split('.')[-2])
            if type == 1:
                type = 0
            if type == 2 or type == 3:
                type = 1
            elif type == 4 or type == 5:
                type = 2
        return type

    def targetTypeImg(file):
        # SiW : SubjectID_SensorID_TypeID_MediumID_SessionID 
        # Type 1 : Live / Type 2 : Image attack / Type 3 : Video attack
        # OULU : Phone_Session_User_Type
        # Type 1 : Live / Type 2,3 : Image attack / Type 4,5 : Video attack

        if file[2] == 'SiW':
            type = int(file[1].split('-')[-4]) - 1
        else:
            type = int(file[1].split('_')[-1].split('-')[-2])
            if type == 1:
                type = 0
            if type == 2 or type == 3:
                type = 1
            elif type == 4 or type == 5:
                type = 2

        return type

    # 把讀取的tensor和target儲存起來
    def saveTensors(path , tensor_list , target_list , dataset , mode , num_frames , scale):
        torch.save(tensor_list , f"{path}/{dataset}_{mode}_{num_frames}_{scale}_tensors.pth")
        print(f"{dataset} {mode} tensor_list for {num_frames} frames is saved successfully!")
        torch.save(target_list , f"{path}/{dataset}_{mode}_{num_frames}_{scale}_targets.pth")
        print(f"{dataset} {mode} target_list for {num_frames} frames is saved successfully!")
        print()
    
    # 印出資訊
    def info(tensor_list , target_list , dataset , mode):
        print(f"# of {dataset} {mode} Tensors : {len(tensor_list)}")
        print(f"# of {dataset} {mode} Targets : {len(target_list)}")
        print(f"Size of First Tensor : {tensor_list[0].shape}")
        print(f"Size of Last  Tensor : {tensor_list[len(tensor_list) - 1].shape}")
        print()

    # 載入儲存好的tensor和target
    def loadTensors(path , dataset , mode , num_frames):
        print(f"Starts to load {dataset} {mode} tensors")
        tensor_list = torch.load(f"{path}/{dataset}_{mode}_{num_frames}_tensors.pth")
        target_list = torch.load(f"{path}/{dataset}_{mode}_{num_frames}_targets.pth")
        print(f"Loaded successfully!")
        return tensor_list ,target_list

    # 印出沒有在最終訓練/測試資料的檔案
    def printStrangeFiles(strange_list):
        print(f"Strange files : ")
        strange_list = sorted(strange_list , key = lambda s: s[1])
        for file in strange_list:
            print(file)
        print(f"# of strange files : {len(strange_list)}")

    # 印出記憶體使用量
    def printMemoryUsage(str):
        print(f"{str}{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.0f} MiB")

    # 載入兩個資料集存好的tensor和target，一起回傳
    def preprocess(save_path , mode , num_frames):
        SiW_tensor_list , SiW_target_list = PreProcess.loadTensors(save_path , "SiW" , mode , num_frames)
        OULU_tensor_list , OULU_target_list = PreProcess.loadTensors(save_path , "OULU" , mode , num_frames)
        tensor_list = OULU_tensor_list + SiW_tensor_list
        target_list = OULU_target_list + SiW_target_list
        return tensor_list , target_list

    # 載入影片並儲存tensor
    def load_and_save(save_path , video_list , face_list , dataset , mode , num_frames , h , w , scale):
        print(f"Starts to load {dataset} {mode} data")
        tensor_list , target_list , strange_list = PreProcess.toTensors(video_list , face_list , num_frames , h , w , scale)
        PreProcess.info(tensor_list  , target_list , dataset , mode)
        PreProcess.saveTensors(save_path , tensor_list , target_list , dataset , mode , num_frames , scale)

        PreProcess.printMemoryUsage("Memory usage after loading : ")
        del tensor_list , target_list
        PreProcess.printMemoryUsage("Memory usage after deletion : ")
        print()
        return strange_list

    def printSize(video_list):
        shapes = []
        for video in tqdm(video_list):
            cap = cv2.VideoCapture(video[0])
            ret, frame = cap.read()             
            if not ret:
               continue
            shapes.append([frame.shape[0] , frame.shape[1]])
        for shape in shapes:
            print(shape)

    def sameNumFrame(video,positions):
        cap = cv2.VideoCapture(video)
        num_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                # print("Complete")
                break
            num_frame = num_frame + 1
        same = num_frame == len(positions)
        if not same: 
            print(f"False Frame Match on {video}")
            print(f"video frames : {num_frame}")
            print(f"face frames : {len(positions)}")
        return same
    
    def toImgList(img_list , path , dataset):
        for img in os.listdir(path):
            img_list.append([path , img , dataset])
        return img_list
    
    
    
    