import torch.nn as nn
from torchvision import models as models

class Classfier(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = models.video.r2plus1d_18()
        self.fc = nn.Sequential(nn.Linear(400 , 3))
    def forward(self, x):
        out = self.res(x)
        return self.fc(out)

class Classfier_r3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = models.video.r3d_18(weights = models.video.R3D_18_Weights.DEFAULT)
        self.fc = nn.Sequential(nn.Linear(400 , 3))
    def forward(self, x):
        out = self.res(x)
        return self.fc(out)

class Classfier_r3d_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = models.video.r3d_18(weights = models.video.R3D_18_Weights.DEFAULT)
        self.fc = nn.Sequential(nn.Linear(400 , 2))
    def forward(self, x):
        out = self.res(x)
        return self.fc(out)

class ImgClassfier_res(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        self.fc = nn.Sequential(nn.Linear(1000 , 512) , nn.ReLU() , nn.Linear(512 , 2))
    def forward(self, x):
        out = self.res(x)
        return self.fc(out)

class ImgClassfier_vgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
        self.fc = nn.Sequential(nn.Linear(1000 , 512) , nn.ReLU() , nn.Linear(512 , 2))
    def forward(self, x):
        out = self.res(x)
        return self.fc(out)


class ImgClassfier_self(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(16, 32, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(32, 64, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 16, 16]

        )
        self.fc = nn.Sequential(
            nn.Linear(64*16*16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    
class ImgClassfier_self2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # [32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [32, 64, 64]

            nn.Conv2d(32, 64, 3, 1, 1), # [64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 32, 32]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 16, 16]
        )
        self.fc = nn.Sequential(
            nn.Linear(128*16*16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    
class ImgClassfier_self4(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(16, 32, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(32, 64, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 16, 16]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 8, 8]
        )
        self.fc = nn.Sequential(
            nn.Linear(128*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class ImgClassfier_self4_np4(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(16, 32, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(32, 64, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 16, 16]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*16*16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class ImgClassfier_self42(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(32, 64, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 16, 16]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 8, 8]

            nn.Conv2d(128, 256, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
        )
        self.fc = nn.Sequential(
            nn.Linear(256*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class ImgClassfier_self43(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(64, 128, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 16, 16]

            nn.Conv2d(128, 256, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            
            nn.Conv2d(256, 512, 3, 1, 1), # [128, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 8, 8]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class ImgClassfier_self52(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(32, 64, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 16, 16]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 8, 8]

            nn.Conv2d(128, 256, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(256, 512, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    
class ImgClassfier_self_np(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(16, 32, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(32, 64, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0),      # [64, 16, 16]
        )
        self.fc = nn.Sequential(
            nn.Linear(64*128*128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    
class ImgClassfier_self_np3(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(16, 32, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(32, 64, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0),      # [64, 16, 16]
        )
        self.fc = nn.Sequential(
            nn.Linear(64*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    
if __name__=='__main__':
    model = models.vgg16()
    print(model)
