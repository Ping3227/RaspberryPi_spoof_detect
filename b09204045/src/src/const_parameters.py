import torch
MODEL_PATH = "../models"
SAVE_PATH = "../tensor_list" # 存tensor的path
TRAIN_IMAGE_PATH = "../images/train"
SIW_TRAIN_IMAGE_PATH = "../images/train/SiW"
SIW_TEST_IMAGE_PATH = "../images/test/SiW"
OULU_TRAIN_IMAGE_PATH = "../images/train/OULU"
OULU_TEST_IMAGE_PATH = "../images/test/OULU"
SIW_TRAIN_PATH = "../../../../capstone_2023/SiW/Train"
SIW_TEST_PATH = "../../../../capstone_2023/SiW/Test"
OULU_TRAIN_PATH = "../../../../capstone_2023/OULU/Train"
OULU_TEST_PATH = "../../../../capstone_2023/OULU/Test"
SELF_REAL_PATH = "../images/self/real"
SELF_FAKE_PATH = "../images/self/fake"
DEVICE =  'cuda' if torch.cuda.is_available else 'cpu'