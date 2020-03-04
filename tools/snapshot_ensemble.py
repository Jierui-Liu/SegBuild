'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-03-01 21:39:47
Description : 
'''
import sys
sys.path.append('..')
from data.dataloader import make_dataloader
from configs import merage_from_arg
from model import bulid_model
from argparse import ArgumentParser
import torch.nn as nn
import torch
from model import bulid_model
torch.backends.cudnn.benchmark = True
import os
from tqdm import tqdm
import cv2 as cv
from configs import merage_from_arg
import datetime
import zipfile

def zipDir(dirpath,outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    for path,dirnames,filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath,'')

        for filename in filenames:
            zip.write(os.path.join(path,filename),os.path.join(fpath,filename))
    zip.close()

def inference_forward(cfg,model_list,dataloader,tta_flag=False):
    print("------- start -------")
    
    device = cfg['device_ids'][0]
    # model = model.cuda(device)
    # model.eval()
    

    with torch.no_grad():
        for image,filename in tqdm(dataloader):
            image = image.cuda(device)
            predict_list = 0
            for model in model_list:
                predict = model(image.clone())

                if tta_flag:
                    predict_flip = model(torch.flip(image,[-1]))
                    predict_flip = torch.flip(predict_flip,[-1])
                    predict += predict_flip
                    predict_flip = model(torch.flip(image,[-2]))
                    predict_flip = torch.flip(predict_flip,[-2])
                    predict_flip += predict_flip
                    predict_flip = model(torch.flip(image,[-1,-2]))
                    predict_flip = torch.flip(predict_flip,[-1,-2])
                    predict += predict_flip
                predict_list += predict
            predict = torch.argmax(predict_list.cpu(),1).byte().numpy()
            
            batch = predict.shape[0]
            save_dir = cfg['save_dir']
            for i in range(batch):
                # import pdb; pdb.set_trace()
                cv.imwrite(os.path.join(save_dir,filename[i].replace('.npy','.tif')),predict[i])
                
                
def load_arg():
    parser = ArgumentParser(description="Pytorch")

    # OUTPUT 
    parser.add_argument("-save_dir",required=True)
    parser.add_argument('-path',required=True,type=str,nargs='+',)
    parser.add_argument('-val_image_dir',type=str)
    parser.add_argument("-device",type=int,nargs='+',
                        help="list of device_id, e.g. [0,1,2]")
    arg = parser.parse_args()
    return arg
    
if __name__ == "__main__":
    # 若更新了load_arg函数，需要对应更新merage_from_arg()
    arg = load_arg()
    save_dir = arg.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dir_list = arg.path


    model_list = []
    for path in dir_list:
        cfg = torch.load(path,map_location='cpu')['cfg']
        if arg.device:
            cfg['device_ids'] = arg.device
        model = bulid_model(cfg['model'],path)
        device = cfg['device_ids'][0]
        model = model.cuda(device)
        model.eval()
        # 默认开启多GPU
        if cfg['multi_gpu']:
            device_ids = cfg['device_ids']
            model = nn.DataParallel(model,device_ids=device_ids)
        model_list.append(model)

    # curr_time = datetime.datetime.now()
    # time_str = datetime.datetime.strftime(curr_time,'%Y%m%d_')
    # model_tag = (os.path.split(arg['MODEL.LOAD_PATH'])[1]).split('.')[0]
    # save_dir = os.path.join(r'../exp/',time_str+model_tag)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print("Save_dir :",save_dir)
    cfg['save_dir'] = save_dir

    if arg.val_image_dir:
        cfg['test_pipeline']['dataset']['test_image_dir'] = arg.val_image_dir

    dataloader = make_dataloader(cfg['test_pipeline'])
    inference_forward(cfg,model_list,dataloader,tta_flag=True)
    zipDir(save_dir,save_dir+".zip") # 压缩至根目录