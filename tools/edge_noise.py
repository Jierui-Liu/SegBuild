import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser
import torch

def load_arg():
    parser = ArgumentParser(description="Pytorch  ")

    # OUTPUT 
    parser.add_argument("-save_dir",required=True)
    arg = parser.parse_args()
    return arg

if __name__ == "__main__":
    torch.set_num_threads(7)
    arg = load_arg()
    print(arg)
    save_dir = arg.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dir_list = [
        r"../exp/new/",
       ]
    for dir in dir_list:
        print(dir,len(os.listdir(dir)))
        assert len(os.listdir(dir))==11481
    file_list = os.listdir(dir_list[0])

    
    def fun(save_dir,dir_list,tif_file):
        tif_list = [cv.imread(os.path.join(dir,tif_file),cv.IMREAD_UNCHANGED) for dir in dir_list]
        tif_list = np.array(tif_list)
        tif_list=tif_list[0]
        
        ks=3
        kernel = np.ones((ks,ks))
        mask = np.zeros((ks,ks))
        edge_acive=cv.filter2D(tif_list, -1, kernel)
        edge_acive[tif_list==ks**2]=0
        edge_acive[tif_list>0]=1
        rand_matrix=np.random.random(tif_list.shape())
        edge_acive=edge_acive*rand_matrix

        tif=tif_list.copy()
        mask[edge_avtive>0.5]=1
        tif=tif-mask
        tif[tif<0]=1
        
        print(tif.shape())
        cv.imwrite(os.path.join(save_dir,tif_file),tif)

    P = Pool(16)
    for tif_file in tqdm(file_list):
        # P.apply_async(fun,(save_dir,dir_list,tif_file))
        tif_list = cv.imread(os.path.join(dir,tif_file),cv.IMREAD_UNCHANGED)
        
        ks=3
        kernel = np.ones((ks,ks))
        mask = np.zeros(tif_list.shape)
        edge_acive=cv.filter2D(tif_list, -1, kernel)
        edge_acive[edge_acive==ks**2]=0
        edge_acive[edge_acive>0]=1
        rand_matrix=np.random.random(tif_list.shape)
        edge_acive=edge_acive*rand_matrix

        tif=tif_list.copy()
        mask[edge_acive>0.5]=1


        tif=tif-mask
        tif[tif<0]=0
        tif[tif>0]=1
        kernel=np.zeros((ks,ks))
        kernel[int(ks//2),:]=1
        kernel[:,int(ks//2)]=1
        tif=cv.filter2D(tif, -1, kernel)
        
        tif[tif<3]=0
        tif[tif>=3]=1
        tif=tif.astype(np.uint8)
        
        cv.imwrite(os.path.join(save_dir,tif_file),tif)

        
    P.close()
    P.join()


        
