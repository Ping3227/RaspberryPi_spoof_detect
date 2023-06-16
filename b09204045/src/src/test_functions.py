from preprocess import PreProcess as pp
import os
path = "../images/opencv/videos"
imgs = []
for x in os.listdir(path):
    imgs.append(os.path.join(path , x )) 

for file in imgs:
    print(file)
    print(pp.targetTypeImg([file , "SiW"]))