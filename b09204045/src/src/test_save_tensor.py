import torch
from tqdm import tqdm
from dataset import VideoDataSet
from torch.utils.data import DataLoader
from preprocess import PreProcess as pp
import torchvision
from torchvision import models

main_path = "../opencv/videos"
SiW_Train_path = "../../../../capstone_2023/SiW/Train"
SiW_Test_path = "../../../../capstone_2023/SiW/Test"
OULU_Train_path = "../../../../capstone_2023/OULU/Train"
OULU_Test_path = "../../../../capstone_2023/OULU/Test"
num_frames = 1

# train_video_list , train_face_list = pp.read_files("" , OULU_Train_path)
test_video_list , test_face_list = pp.read_files("" , OULU_Test_path)
train_video_list , train_face_list = pp.read_files(main_path,main_path)

device = 'cuda' if torch.cuda.is_available else 'cpu'
print(f"Device : {device}")
print(f"Frames : {num_frames}")
false = pp.num_mismatch(train_video_list , train_face_list)    
# print(f"# of Videos : {len(train_video_list)}")
# print(f"# of Faces  : {len(train_face_list)}")
print(f"# of Mismatch : {false}")
model = models.video.r2plus1d_18().to(device)

print("Starts to load train data")
train_tensor_list , train_target_list , train_strange_list = pp.toTensors(train_video_list , train_face_list , num_frames , 128 , 171 , 0.9)

print("Starts to load test data")
test_tensor_list , test_target_list , test_strange_list = pp.toTensors(test_video_list , test_face_list , num_frames , 128 , 171 , 0.9)

print(f"# of Train Tensors : {len(train_tensor_list)}")
print(f"# of Train Targets : {len(train_target_list)}")
print(f"# of Test Tensors : {len(test_tensor_list)}")
print(f"# of Test Targets : {len(test_target_list)}")
print(f"Size of First Train Tensor : {train_tensor_list[0].shape}")
print(f"Size of Last Train Tensor : {train_tensor_list[len(train_tensor_list) - 1].shape}")
# print(f"Size of Test Tensors : {test_tensor_list[0].shape}")
strange_list = train_strange_list + test_strange_list
print("Strange files:")
for file in strange_list:
    print(file)
print(len(strange_list))

torch.save(train_tensor_list , "train_tensor_list.pth")
saved_tensor_list = torch.load("train_tensor_list.pth")
same = True

for c in range(len(train_tensor_list[0])):
    for f in range(len(train_tensor_list[0][c])):
        for h in range(len(train_tensor_list[0][c][f])):
            for w in range(len(train_tensor_list[0][c][f][h])):
                same = True
                if train_tensor_list[0][c][f][h][w] != saved_tensor_list[0][c][f][h][w]:
                    same = False
                    print(f"{c} {f} {h} {w}")
print(same)