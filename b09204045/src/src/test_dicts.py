import torch
import os
import const_parameters as cp

# # self
# dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 128519 epoch 2 batch 256 lr 1e-4 wd 1e-5 self.pth")) # 69% ipad 錯2
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 129778 batch 256 lr 5e-05.pth"))
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 126808 batch 2048 lr 0.0001.pth"))

# # self bright 0.5 other 0
# dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 122673 0.952 batch 256 lr 0.0001.pth")) # 65% ipad 錯3 覺得吳海源是人 超爛
# dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 125518 batch 256 lr 0.0001.pth")) # 73% ipad 錯3 爭議的兩個我都猜錯

# # self bright 0.5 other 0.1
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 119248 0.943 batch 256 lr 0.0001.pth")) # 73% ipad 錯2
# DICT = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 124994 0.947 batch 256 lr 0.0001.pth")) # 80% ipad 錯2 可以分爭議的兩個我 目前最準
# # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 129000 0.947 batch 256 lr 0.0001.pth")) # 76% ipad 錯3 覺得吳海源是人

# self bright 0.5 other 0.1 translate 0.05
# DICT = torch.load(os.path.join(cp.MODEL_PATH ,"img/ImgClassfier_self self acc 0.7376 test acc 0.9626 batch 256 lr 0.0001.pth")) # 66/79/74  爭議錯

# self bright 0.5 other 0.2
# DICT = torch.load(os.path.join(cp.MODEL_PATH ,"img/ImgClassfier_self self acc 0.7186 test acc 0.9389 batch 256 lr 0.0001.pth")) # 83/72/76 爭議錯

# # # self_np3 bright 0.5 other 0.1
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 127211 0.945 batch 256 lr 0.0001.pth")) # 80% ipad 錯2 爭議的兩個我都覺得是真的
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.954 batch 256 lr 0.0001.pth")) # 73% ipad 錯2 爭議的兩個我都覺得是真的
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 129000 0.947 batch 256 lr 0.0001.pth")) # 76% ipad 錯3 覺得吳海源是人

# # self_np bright 0.5 other 0.1
# dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.942 batch 256 lr 0.0001.pth")) # 57% ipad 錯3

# # # self2 bright 0.5 other 0.2
# # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 130717 batch 256 lr 0.0001.pth")) # 65% # 69% ipad錯4 覺得手幅是人
# # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 127976 0.937 batch 256 lr 0.0001.pth")) # 76% ipad錯3
# # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.937 batch 256 lr 0.0001.pth")) # 65% /
# # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.951 batch 256 lr 0.0001.pth")) # 80% / 76%  self 2 覺得jordon是人 ipad錯4
# # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.958 batch 256 lr 0.0001.pth")) # 71% / 76%  self 2

# # # self2 bright 0.5 other 0.1
# DICT = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 130649 0.932 batch 256 lr 0.0001.pth")) # 80% ipad 錯2 沒辦法分爭議的兩個我
# # # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 131047 0.948 batch 256 lr 0.0001.pth")) # 73%
# # # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.958 batch 256 lr 0.0001.pth")) # 73% ipad 錯2
# # # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.961 batch 256 lr 0.0001.pth")) # 76% ipad 錯2 覺得手幅是人

# # # self2 bright 0.6 other 0.1
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 121122 0.934 batch 256 lr 0.0001.pth")) # 73% ipad 錯2 會分爭議的兩個我 覺得手幅和吳海嫄是人
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 130250 batch 256 lr 0.0001.pth")) # 69% ipad 錯3 不會分爭議的兩個我 覺得手幅是人
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.941 batch 256 lr 0.0001.pth")) # 76% ipad 錯3 會分爭議的兩個我 覺得吳海嫄是人
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.952 batch 256 lr 0.0001.pth")) # 69% ipad 錯4 不會分爭議的兩個我 覺得手幅是人

# # # # self2 bright 0.5 other 0.15
# # # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 129141 0.936 batch 256 lr 0.0001.pth")) # 69% ipad錯4 覺得手幅是人
# # # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.947 batch 256 lr 0.0001.pth")) # 76% ipad錯3  覺得jordon是人
# # # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 130961 batch 256 lr 0.0001.pth")) # 69% ipad錯4

# # # self4 bright 0.5 other 0.1 都覺得前兩個錯
# DICT = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 123230 0.948 batch 256 lr 0.0001.pth")) # 76% ipad 錯1 會分爭議的兩個我 覺得吳海嫄是人
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 128489 batch 256 lr 0.0001.pth")) # 73% ipad 錯1 會分爭議的兩個我 覺得吳海嫄和志淇是人
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.955 batch 256 lr 0.0001.pth")) # 76% ipad 錯1 覺得爭議的兩個我都錯
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 131382 batch 256 lr 0.0001.pth")) # 65% ipad 錯3 覺得爭議的兩個我都對 覺得吳海嫄是人

# # self42 bright 0.5 other 0.1 都覺得前兩個錯
# dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 132344 0.938 batch 256 lr 0.0001.pth")) # 73% ipad 錯3 會分爭議的兩個我
DICT = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.966 batch 256 lr 0.0001.pth")) # 76% / 80% ipad 錯1 會分爭議的兩個我
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.970 batch 256 lr 0.0001.pth")) # 80% ipad 錯2 會分爭議的兩個我

# # self43 bright 0.5 other 0.1
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.927 batch 256 lr 0.0001.pth")) # 69% 爛死
# DICT = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.956 batch 256 lr 0.0001.pth")) # 84% ipad 錯2
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.960 batch 256 lr 0.0001.pth")) # 76% ipad 錯4

# # # self4_np4 bright 0.5 other 0.1
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 123030 0.949 batch 256 lr 0.0001.pth")) # 76% ipad 錯1 爭議都覺得是假的
# # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 0.955 batch 256 lr 0.0001.pth")) # 80% ipad 錯1 爭議都覺得是假的

# self52 bright 0.5 other 0.1
#DICT = torch.load(os.path.join(cp.MODEL_PATH ,"img/ImgClassfier_self52 self acc 0.7709 test acc 0.9652 batch 256 lr 0.0001.pth")) # 
# # # # vgg16 bright 0.5 other 0.1
# # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 126912 0.968 batch 256 lr 0.0001.pth"))  # 73% ipad錯2 感覺還是爛爛的
# # # dict = torch.load(os.path.join(cp.MODEL_PATH ,"img/151 frames 132197 0.975 batch 256 lr 0.0001.pth"))  # 97% / 69% ipad錯3 overfitting 遠人都覺得是假的

 