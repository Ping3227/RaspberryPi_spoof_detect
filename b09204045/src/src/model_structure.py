from torchvision import models
import torch.nn as nn

model = models.video.r2plus1d_18()
model.add_module("added" , nn.Linear(32 , 3))
print(model)