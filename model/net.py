import torch
import torch.nn as nn
import sys
sys.path.append('..')
from model.HRNet_ocr.models.seg_hrnet_ocr import get_seg_model
from model.HRNet_ocr.hrnet_ocr_config import config

class hrnet_w48_ocr_up4(nn.Module):
    def __init__(self,num_classes=2,freeze_num_layers=0):
        super(hrnet_w48_ocr_up4,self).__init__()
        self.num_classes = num_classes
        self.freeze_num_layers = freeze_num_layers

        config.merge_from_file(r"../model/HRNet_ocr/hrnet_ocr_config/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml")
        self.hrnet_w48_ocr = get_seg_model(config)
        state_dict = self.hrnet_w48_ocr.state_dict()
        
        # pretrain model
        pretrain_state_dict = torch.load(r"../model/HRNet_ocr/models/pretrain_model/hrnet_ocr_cs_trainval_8227_torch11.pth",map_location='cpu')
        keys = list(pretrain_state_dict.keys())

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for item in keys:
            if item[6:] in state_dict.keys() and 'cls_head' not in item and 'aux_head.3' not in item:
                new_state_dict[item[6:]] = pretrain_state_dict.pop(item)
        state_dict.update(new_state_dict)
        self.hrnet_w48_ocr.load_state_dict(state_dict)

        self.up2 = nn.Upsample(scale_factor=2)
        self.up4 = nn.Upsample(scale_factor=4)

        
        self.aux_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=16,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1)
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1)
        )


        self.freeze_layers(num_layers=self.freeze_num_layers)

    def forward(self,x):
        # print('in ',x.shape)
        x = self.hrnet_w48_ocr(x)
        
        x1 = self.up4(x[0])
        x1 = self.aux_layer(x1)
        # print('out 1',x1.shape)
        x2 = self.up4(x[1])
        x2 = self.last_layer(x2)
        # print('out 2',x2.shape)
        return [self.up2(x1),self.up2(x2)]    

    def freeze_layers(self,num_layers=9): #默认冻结前9层
        if num_layers <= 0:
            pass
        else:
            for i,(name,child) in enumerate(self.hrnet_w48_ocr.named_children()):
                if i < num_layers:
                    print("freeze layer : ",name)
                    for param in child.parameters():
                        param.requires_grad = False
                        i += 1


if __name__ == "__main__":
    
    # model = hrnet_w18()
    # inputs = torch.randn(1,3,256,256)
    # outputs = model(inputs)
    # print(outputs.shape)

    # model = hrnet_w48_up4()
    # inputs = torch.randn(1,3,256,256)
    # outputs = model(inputs)
    # print(outputs.shape)
    pass

