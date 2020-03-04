import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser

def load_arg():
    parser = ArgumentParser(description="Pytorch")

    # OUTPUT 
    parser.add_argument("-save_dir",required=True)
    parser.add_argument('-path',required=True,type=str,nargs='+',)
    arg = parser.parse_args()
    return arg

if __name__ == "__main__":
    arg = load_arg()
    print(arg)
    save_dir = arg.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    dir_list = arg.path
    print(dir_list)
    if len(dir_list)%2==0:
        print("Warning : 投票模型数目最好为偶数 ")
    for dir in dir_list:
        assert len(os.listdir(dir))==11481
    file_list = os.listdir(dir_list[0])


    def fun(save_dir,dir_list,tif_file):
        len_dir_list = len(dir_list)
        tif_list = [cv.imread(os.path.join(dir,tif_file),cv.IMREAD_UNCHANGED) for dir in dir_list]
        tif_list = np.array(tif_list)
        tif = tif_list.sum(axis=0)
        tif[tif<=(len_dir_list-1)//2] = 0
        tif[tif> (len_dir_list-1)//2] = 1
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


        
