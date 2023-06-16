from tqdm import tqdm
from dataset import ImgDataSet
from torch.utils.data import DataLoader
from preprocess import PreProcess as pp
from model import ImgClassfier_res
from model import ImgClassfier_self
from model import ImgClassfier_self2
from model import ImgClassfier_self4
import numpy as np
import const_parameters as cp
import torch.nn as nn
import torch
import os

myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# 超參數
num_frames = 151
batch_size = 256
num_epoch = 3
device = cp.DEVICE
learning_rate = 1e-4 # 1e-4
weight_decay = 1e-5 # 1e-5
model = ImgClassfier_self2()
model.to(device)
print(f"Device : {device}")
print(f"Frames : {num_frames}")
print(f"Batch : {batch_size}")
print(f"Epoch : {num_epoch}")
print(f"Learning Rate : {learning_rate}")

pp.printMemoryUsage("Memory Usage before loading : ")
train_file_list = []
train_file_list = pp.toImgList(train_file_list , cp.OULU_TRAIN_IMAGE_PATH ,"OULU")
train_file_list = pp.toImgList(train_file_list , cp.SIW_TRAIN_IMAGE_PATH ,"SiW")

test_file_list = []
test_file_list = pp.toImgList(test_file_list , cp.OULU_TEST_IMAGE_PATH ,"OULU")
test_file_list = pp.toImgList(test_file_list , cp.SIW_TEST_IMAGE_PATH ,"SiW")

train_set    = ImgDataSet(train_file_list)
train_loader = DataLoader(train_set , batch_size = batch_size , shuffle = True)

test_set    = ImgDataSet(test_file_list)
test_loader = DataLoader(test_set , batch_size = batch_size , shuffle = False)

pp.printMemoryUsage("Memory Usage after loading : ")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate , weight_decay = weight_decay )

print("Starts to train")
best_acc = 0
best_tp = 0
train_size = len(train_set)
test_size = len(test_set)
for epoch in range(num_epoch):
    model.train()
    train_loss = 0.0
    train_acc = 0
    true_positive = 0 # 把真的認成真的
    false_positive = 0 # 把假的認成真的
    true_negative = 0 # 把假的認成假的
    false_negative = 0 # 把真的認成假的
    acc_change = False
    tp_change = False
    for batch in tqdm(train_loader):
        imgs , targets = batch
        outputs = model(imgs.to(device))
        loss = criterion(outputs , targets.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 找出最值最大的標籤
        _ , pred = torch.max(outputs , dim = 1)

        # 加上正確數量
        train_acc += (pred == targets.to(device)).sum().item()
        train_loss += loss.item()

    model.eval()
    test_loss = 0.0
    test_acc = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            imgs , targets = batch
            outputs = model(imgs.to(device))
            loss = criterion(outputs , targets.to(device))

            # 找出最值最大的標籤
            _ , pred = torch.max(outputs , dim = 1)
            # 加上正確數量
            test_acc += (pred == targets.to(device)).sum().item()
            test_loss += loss.item()

            for i in range(len(pred)):
                if pred[i] == 0: 
                    if targets[i] == 0: # 把真的認成真的
                        true_positive += 1
                    else: # 把假的認成真的
                        false_positive += 1
                else: 
                    if targets[i] == 0: # 把真的認成假的
                        false_negative += 1
                    else: # 把假的認成假的
                        true_negative += 1
   
    true_positive_rate = true_positive/test_size
    true_negative_rate = true_negative/test_size
    false_positive_rate = false_positive/test_size
    false_negative_rate = false_negative/test_size

    print(f'[{epoch+1:03d}/{num_epoch:03d}] Train Acc: {train_acc/len(train_set):3.5f} Loss: {train_loss/len(train_loader):3.5f} | Val Acc: {test_acc/len(test_set):3.5f} loss: {test_loss/len(test_loader):3.5f}')
    print(f"True Positive: {true_positive_rate:3.5f} False Positive: {false_positive_rate:3.5f}")
    print(f"True Negative: {true_negative_rate:3.5f} False Negative: {false_negative_rate:3.5f}")
    # if the model improves, save a checkpoint at this epoch


    if train_acc/len(train_set) >= test_acc/len(test_set):

        if true_positive > best_tp:
            best_tp = true_positive
            tp_change = True
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_rate = best_acc/len(test_set)
            acc_change = True
    
    if tp_change == True:
        if acc_change == True:
            torch.save(model.state_dict(), os.path.join(cp.MODEL_PATH , f"img/{num_frames} frames {best_tp} {best_acc_rate:.3f} batch {batch_size} lr {learning_rate}.pth"))
            print(f'saving model with best_tp {best_tp} and acc {best_acc_rate:5f}')
        else:
            torch.save(model.state_dict(), os.path.join(cp.MODEL_PATH , f"img/{num_frames} frames {best_tp} batch {batch_size} lr {learning_rate}.pth"))
            print(f'saving model with best_tp {best_tp}')
    else:
        if acc_change == True:
            torch.save(model.state_dict(), os.path.join(cp.MODEL_PATH , f"img/{num_frames} frames {best_acc_rate:.3f} batch {batch_size} lr {learning_rate}.pth"))
            print(f'saving model with acc {best_acc_rate:.5f}')

   