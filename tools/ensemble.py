import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser

def load_arg():
    parser = ArgumentParser(description="Pytorch  ")

    # OUTPUT 
    parser.add_argument("-save_dir",required=True)
    arg = parser.parse_args()
    return arg

if __name__ == "__main__":
    arg = load_arg()
    print(arg)
    save_dir = arg.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir('../exp/gray'):
        os.makedirs('../exp/gray')

    dir_list = [
        # r"/home/chenbangdong/cbd/LinHonghui/exp/20200225_HRNet_W48_p7794/epoch_4", #78.14
        # r"../exp/result_0.7860/",
        # r"../exp/result_0.8074/",
        r"../exp/20200306_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_24000/",
        r"../exp/20200306_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_32000/",
        r"../exp/20200306_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_40000/",
        # r"../exp/20200305_hrnet_w48_ocr_up4_default_w8_lr1_50000/",
        # r"/home/l/deeplearning_1/cbd/LinHonghui/SegBulid_jr/result_0.7682_labelsmooth_epoch4",# 
        # r"/home/l/deeplearning_1/cbd/LinHonghui/SegBuild_master/exp/20200303_hrnet_w48_ocr_up4_configs_20000"# 不确定
    ]
    for dir in dir_list:
        assert len(os.listdir(dir))==11481
    file_list = os.listdir(dir_list[0])

    rate=0.97
    num_pixels=1024*1024
    thres=rate*num_pixels
    delete=['1ee2da','2a8a2c','2b1313','6a7a3a','9b424f','57abb4','597db5','621d7d','625a04','671d22','849fa9','912db2',\
        '1982d0','4136dd','8998f4','52743c','a515a5','ab7a66','abaaf1','af21c0','affdc4','cca260']
    def fun(save_dir,dir_list,tif_file):
        tif_list = [cv.imread(os.path.join(dir,tif_file),cv.IMREAD_UNCHANGED) for dir in dir_list]
        tif_list = np.array(tif_list)
        
        tif = tif_list.sum(axis=0)
        tif[tif<2] = 0
        tif[tif>=2] = 1
        


        cv.imwrite(os.path.join(save_dir,tif_file),tif)

    P = Pool(16)
    for tif_file in file_list:
        P.apply_async(fun,(save_dir,dir_list,tif_file))
        # tif_list = [cv.imread(os.path.join(dir,tif_file),cv.IMREAD_UNCHANGED) for dir in dir_list]
        # tif_list = np.array(tif_list)
        # tif = tif_list.sum(axis=0)
        # tif[tif<2] = 0
        # tif[tif>=2] = 1
        # cv.imwrite(os.path.join(save_dir,tif_file),tif)
    P.close()
    P.join()


        
