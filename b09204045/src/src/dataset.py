from torch.utils.data import Dataset
from preprocess import PreProcess as pp
from PIL import Image
import os
import cv2
import numpy as np
import torch
import const_parameters as cp


class VideoDataSet(Dataset):
    def __init__(self , tensor_list , target_list):
        super().__init__()
        self.tensor_list = tensor_list
        self.target_list = target_list
        self.size = len(tensor_list)
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        if self.target_list[index] == 2:
            self.target_list[index] = 1
        return self.tensor_list[index] , self.target_list[index]

class ImgDataSet(Dataset):
    def __init__(self , file_list):
        super().__init__()
        self.file_list = file_list
        self.size = len(file_list)
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        root = self.file_list[index][0]
        file_name = self.file_list[index][1]
        dataset = self.file_list[index][2]
        file_path = os.path.join(root,file_name)

        if dataset == "self":
            if root == cp.SELF_REAL_PATH:
                target = 0
            else:
                target = 1
        else:
            target = pp.targetTypeImg(self.file_list[index])

        img = cv2.imread(file_path) # BGR
        img = np.transpose(img , (2,0,1)) # (H,W,C) -> (C,H,W)
        img = torch.tensor(img , dtype = torch.float32)
       
        if target == 2:
            target = 1
        
        return img , target
    
class ImgDataSet_Test(Dataset):
    def __init__(self , face_list):
        super().__init__()
        self.face_list = face_list
        self.size = len(face_list)
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        return self.face_list[index]
    
class ImgDataSet_Aug(Dataset):
    def __init__(self , file_list , tfm):
        super().__init__()
        self.file_list = file_list
        self.size = len(file_list)
        self.tfm = tfm
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        root = self.file_list[index][0]
        file_name = self.file_list[index][1]
        file_path = os.path.join(root,file_name)

        target = pp.targetTypeImg(self.file_list[index])

        img = Image.open(file_path) 
        img = self.tfm(img) # 做transform
        img = np.array(img) # RGB
        img = cv2.cvtColor(img , cv2.COLOR_RGB2BGR) # 變成BGR
        # cv2.imshow("face", img)
        # cv2.waitKey(100)
        img = np.transpose(img , (2,0,1)) # (H,W,C) -> (C,H,W)
        img = torch.tensor(img , dtype = torch.float32)
       
        if target == 2:
            target = 1
        
        return img , target