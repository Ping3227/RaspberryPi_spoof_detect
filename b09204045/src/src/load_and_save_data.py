from preprocess import PreProcess as pp
import const_parameters as cp
main_path = "../opencv/videos"
local_save_path = "."
num_frames = 30
h = 128
w = 171
scale = 0.9
mode = "test" # all: 全部 
num_mismatch = 0

if mode == "all" or mode == "SiW":
    SiW_train_video_list , SiW_train_face_list = pp.readSiW(cp.SIW_TRAIN_PATH)
    SiW_test_video_list , SiW_test_face_list   = pp.readSiW(cp.SIW_TEST_PATH)
    train_mismatch = pp.num_mismatch(SiW_train_video_list , SiW_train_face_list)
    test_mismatch = pp.num_mismatch(SiW_test_video_list , SiW_test_face_list)
    num_mismatch = num_mismatch + train_mismatch + test_mismatch

if mode == "all" or mode == "OULU":
    OULU_train_video_list , OULU_train_face_list = pp.readOULU(cp.OULU_TRAIN_PATH)
    OULU_test_video_list , OULU_test_face_list   = pp.readOULU(cp.OULU_TEST_PATH)
    train_mismatch = pp.num_mismatch(OULU_train_video_list , OULU_train_face_list)
    test_mismatch = pp.num_mismatch(OULU_test_video_list , OULU_test_face_list)
    num_mismatch = num_mismatch + train_mismatch + test_mismatch

if mode == "test":
    train_video_list , train_face_list = pp.read_files(main_path,main_path)
    test_video_list , test_face_list   = pp.read_files(main_path,main_path)

print(f"Mode : {mode}")
print(f"Frames : {num_frames}") 
print(f"# of Mismatch : {num_mismatch}")
print()
strange_list = []

if(mode == "all" or mode == "OULU"):
    OULU_train_strange_list = pp.load_and_save(cp.SAVE_PATH,OULU_train_video_list,OULU_train_face_list,"OULU","train",num_frames,h,w,scale)
    OULU_test_strange_list = pp.load_and_save(cp.SAVE_PATH,OULU_test_video_list,OULU_test_face_list,"OULU","test",num_frames,h,w,scale)
    strange_list = OULU_train_strange_list + OULU_test_strange_list

if(mode == "all" or mode == "SiW"):
    SiW_train_strange_list = pp.load_and_save(cp.SAVE_PATH,SiW_train_video_list,SiW_train_face_list,"SiW","train",num_frames,h,w,scale)
    SiW_test_strange_list = pp.load_and_save(cp.SAVE_PATH,SiW_test_video_list,SiW_test_face_list,"SiW","test",num_frames,h,w,scale)
    strange_list = strange_list + SiW_train_strange_list + SiW_test_strange_list

if(mode == "test"):
    strange_list = pp.load_and_save(local_save_path,train_video_list,train_face_list,"OULU","train",num_frames,h,w,scale)
pp.printStrangeFiles(strange_list)