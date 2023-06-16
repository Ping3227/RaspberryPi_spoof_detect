import torchvision.transforms as tf
import numpy as np
import cv2
from PIL import Image
img = Image.open("../opencv/test_videos/results/fake-0-22.jpg")
cj = tf.ColorJitter(brightness = 0.5 , contrast = 0.1 , saturation = 0.1)
hf = tf.RandomHorizontalFlip(p = 0.5)
ra = tf.RandomAffine(degrees = 0 , translate = (0.05 , 0.05)) # 旋轉 + 位移
tfm = tf.Compose([cj,ra,hf])
print(tfm)

img = tfm(img)
img.show()

# # PIL Image 讀進來會是RGB
# img_np = np.array(img)
# print(img_np)
# print(img_np.shape)

# # CV2看到的是BGR 所以紅藍對調了
# cv2.imshow("pil to array",img_np)
# cv2.waitKey(1000)

# # 把img_的channel 改成 BGR
# img_t = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
# cv2.imshow("pil to array",img_t)
# cv2.waitKey(1000)