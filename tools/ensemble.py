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
        r"../exp/result_0.7860/",
        # r"../exp/result_0.8074/",
        # r"../exp/20200308_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_3_24000/",
        # r"../exp/20200308_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_3_32000/",

        r"../exp/20200313_hrnet_w48_ocr_up4_default_4_pesudo_highscore_2_continue_4_16000/",
        r"../exp/20200313_hrnet_w48_ocr_up4_default_4_pesudo_highscore_2_continue_4_24000/",
        r"../exp/result_8324/",
        # r"../exp/20200307_hrnet_w48_ocr_up4_default_w95_pesudo_highrecall_1_24000/",
        # r"../exp/20200307_hrnet_w48_ocr_up4_default_w95_pesudo_highrecall_1_32000/",
        # r"../exp/20200307_hrnet_w48_ocr_up4_default_w95_pesudo_highrecall_1_40000/",
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
        tif_mask=tif_list[0,:,:]
        
        if tif_mask.sum()==0:
            tif = tif_list.sum(axis=0)
            
            image_init=cv.imread(os.path.join('/home/liujierui/proj/Dataset/test',tif_file[:-4],tif_file))
            image_init_r=image_init[:,:,2]
            mean_r=image_init_r.mean()
            mean_bgr=image_init.mean()
            image_init_b=image_init[:,:,0]
            image_init_g=image_init[:,:,1]
            mean_g=image_init_g.mean()
            temp1=image_init_r-image_init_g
            temp2=image_init_g-image_init_b
            temp3=np.zeros_like(image_init_g)
            temp3[image_init_g>230]=1
            temp3[image_init_g<50]=1
            temp4=temp1+temp2+temp3
            indexs_x=np.where(np.abs(temp4)<=2)[0]
            if len(indexs_x)>10000 and mean_g<1.02*mean_bgr and mean_r<1.08*mean_bgr:# and delta_g.sum()<num_pixels*10:
                print('gray:',tif_file)
                cv.imwrite('../exp/gray/'+tif_file,image_init)
                tif[tif>=1] = 1
                # tif[tif<2] = 0
                # tif[tif>=2] = 1
            else:
                # print('not gray:',tif_file)
                tif[tif>=0] = 0
        else:
            tif_list=tif_list[1:,:,:]
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


        
