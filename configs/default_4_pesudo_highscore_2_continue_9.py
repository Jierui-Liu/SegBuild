'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-02-29 22:24:22
Description : 
'''

config = dict(
    # Basic Config
    log_period = 0.01,
    tag = "",
    find_lr = False,
    max_epochs = 24 + 1,
    # save_per_step = 100, # 最好是设置为 swa 中 step_size_up 的偶数倍数
    save_dir = r"../output/model/",
    enable_backends_cudnn_benchmark = True,
    # Dataset
    train_pipeline = dict(
        transforms = [
                    dict(type="RandomStainNorm",p=0.1), 
                    dict(type="RandomCrop",p=1,output_size=(1024,1024)),
                    dict(type="RandomHorizontalFlip",p=0.5),
                    dict(type="RandomVerticalFlip",p=0.5),
                    dict(type="ColorJitter",brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
                    dict(type="Shift_Padding",p=0.1,hor_shift_ratio=0.05,ver_shift_ratio=0.05,pad=0),
                    dict(type="RandomErasing",p=0.2,sl=0.02,sh=0.4,rl=0.2),
                    dict(type="GaussianBlur",p=0.2,radiu=2),
                    dict(type="Rescale",output_size=(512,512)),
                    dict(type="ToTensor",),
                    dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False),
                    ],
        dataset = dict(type="load_dataAll",
                    # train_1
                    train_1_image_dir = r'/home/liujierui/proj/Dataset/dataset1024/image_bin',train_1_mask_dir = r'/home/liujierui/proj/Dataset/dataset1024/label_bin', 
                    # train_2
                    # train_2_image_dir = r"/home/chenbangdong/cbd/DrivenDATA/train_tier2_dataset/image_bin",train_2_mask_dir = r"/home/chenbangdong/cbd/DrivenDATA/train_tier2_dataset/label_bin", 
                    train_2_image_dir = r"",train_2_mask_dir = r"", 
                    # pseudo_label
                    pseudo_image_dir = r"/home/liujierui/proj/Dataset/test_dataset_npy/image_bin", pseudo_mask_dir = r"/home/liujierui/proj/SegBuild_master/exp/f/ff",  
                    # pseudo_image_dir = r"", pseudo_mask_dir = r"",  
                    # extra_datasets --> List[]
                    # extra_image_dir_list = [r"/home/liujierui/proj/Dataset/AerialImageDataset/image_bin"],  extra_mask_dir_list = [r"/home/liujierui/proj/Dataset/AerialImageDataset/label_bin"],   
                    extra_image_dir_list = [r""],  extra_mask_dir_list = [r""],  
        ),
        batch_size = 4,
        shuffle = True,
        num_workers = 8,
        drop_last = True
    ),

    test_pipeline = dict(
        transforms = [
                    dict(type="RandomCrop",p=0.1,output_size=(1024,1024)),
                    dict(type="Rescale",output_size=(512,512)),
                    dict(type="ToTensor",),
                    dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False),
        ],
        dataset = dict(type="load_Inference_dataset",
                    test_image_dir = r"/home/liujierui/proj/Dataset/test_dataset_npy/image_bin/"
                    ),
        batch_size = 16,
        shuffle = False,
        num_workers = 8,
        drop_last = False,  
    ),


    # Model
    model = dict(
        type = "hrnet_w48_ocr_up4",
        num_classes = 2,
        freeze_num_layers = 7,
    ),
    # pretrain = r"",
    # pretrain = r"/home/liujierui/proj/SegBuild_master/model/HRNet_ocr/models/pretrain_model/f_1025rescale512_epoch_6__hrnet_w48_ocr.pth",         #file/path of the pretrain model
    pretrain = r"/home/liujierui/proj/SegBuild_master/output/model/hrnet_w48_ocr_up4_default_4_pesudo_highscore_2_continue_8_36000.pth",         #file/path of the pretrain model
    multi_gpu = True,
    device_ids = [1], # 默认第一位作为主卡

    # Solver 
    enable_swa = False,
    criterion = dict(type="CrossEntropy_ocr", num_outputs = 2,balance_weights=[0.1,1.3]),
    lr_scheduler = dict(type="CyclicLR",base_lr=1e-6,max_lr=(1e-2)/5,step_size_up=4000,mode='triangular2',cycle_momentum=True), # cycle_momentum=False if optimizer==Adam
    optimizer = dict(type="SGD",lr=1e-4,momentum=0.9,weight_decay=1e-5),

    

)


if __name__ == "__main__":

    pass