from preprocess import PreProcess as pp
import const_parameters as cp
main_path = "../opencv/videos"
local_save_path = "."
num_frames = 151
h = 128
w = 128
scale = 0.9
mode = "all" # all: 全部 
num_mismatch = 0
train_index = 0
test_index = 0

if mode == "all" or mode == "OULU":
    OULU_train_video_list , OULU_train_face_list = pp.readOULUImg(cp.OULU_TRAIN_PATH)
    OULU_test_video_list , OULU_test_face_list   = pp.readOULUImg(cp.OULU_TEST_PATH)
    train_mismatch = pp.num_mismatchImg(OULU_train_video_list , OULU_train_face_list)
    test_mismatch = pp.num_mismatchImg(OULU_test_video_list , OULU_test_face_list)
    num_mismatch = num_mismatch + train_mismatch + test_mismatch

if mode == "all" or mode == "SiW":
    SiW_train_video_list , SiW_train_face_list = pp.readSiWImg(cp.SIW_TRAIN_PATH)
    SiW_test_video_list , SiW_test_face_list   = pp.readSiWImg(cp.SIW_TEST_PATH)
    train_mismatch = pp.num_mismatchImg(SiW_train_video_list , SiW_train_face_list)
    test_mismatch = pp.num_mismatchImg(SiW_test_video_list , SiW_test_face_list)
    num_mismatch = num_mismatch + train_mismatch + test_mismatch

if mode == "test":
    SiW_train_video_list , SiW_train_face_list = pp.readSiWImg(main_path)
    train_mismatch = pp.num_mismatchImg(SiW_train_video_list , SiW_train_face_list)
    num_mismatch = num_mismatch + train_mismatch

    OULU_train_video_list , OULU_train_face_list = pp.readOULUImg(main_path)
    train_mismatch += pp.num_mismatchImg(OULU_train_video_list , OULU_train_face_list)
    num_mismatch = num_mismatch + train_mismatch

# for sv in SiW_train_video_list:
#     print(sv)
# for sf in SiW_train_face_list:
#     print(sf)
# for ov in OULU_train_video_list:
#     print(ov)
# for of in OULU_train_face_list:
#     print(of)

print(f"Mode : {mode}")
print(f"Frames : {num_frames}") 
print(f"# of Mismatch : {num_mismatch}")
print()
strange_list = []

if(mode == "all" or mode == "OULU"):
    OULU_train_strange_list = pp.toImages(cp.OULU_TRAIN_IMAGE_PATH,OULU_train_video_list,OULU_train_face_list,num_frames,h,w,scale,train_index)
    OULU_test_strange_list = pp.toImages(cp.OULU_TEST_IMAGE_PATH,OULU_test_video_list,OULU_test_face_list,num_frames,h,w,scale,test_index)
    strange_list = OULU_train_strange_list + OULU_test_strange_list

if(mode == "all" or mode == "SiW"):
    SiW_train_strange_list = pp.toImages(cp.SIW_TRAIN_IMAGE_PATH,SiW_train_video_list,SiW_train_face_list,num_frames,h,w,scale , train_index)
    SiW_test_strange_list = pp.toImages(cp.SIW_TEST_IMAGE_PATH,SiW_test_video_list,SiW_test_face_list,num_frames,h,w,scale ,test_index)
    strange_list = strange_list + SiW_train_strange_list + SiW_test_strange_list

if(mode == "test"):
    strange_list = pp.toImages( "../images/SiW", SiW_train_video_list,SiW_train_face_list,num_frames,h,w,scale,train_index)
    strange_list += pp.toImages( "../images/OULU", OULU_train_video_list,OULU_train_face_list,num_frames,h,w,scale,train_index)
pp.printStrangeFiles(strange_list)