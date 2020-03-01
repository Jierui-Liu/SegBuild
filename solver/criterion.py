'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-02-29 11:58:50
Description : 
'''
import torch
from torch import nn


class cross_entropy2d(nn.Module):
    def __init__(self):
        super(cross_entropy2d,self).__init__()
    def forward(self,input, target, weight=None, size_average=True):
        if weight:
            weight = torch.tensor(weight,device=target.device)
        # print(input.shape,target.shape)
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        log_p = nn.functional.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        
        loss = nn.functional.nll_loss(log_p, target.long(), weight=weight, reduction='sum')
        if size_average:
            loss /= mask.data.sum()
        return loss

class DataParallel_Loss(nn.Module): # 待修改
    def __init__(self):
        super(DataParallel_Loss,self).__init__()
    def forward(self,input,target):
        return input

if __name__ == "__main__":
    loss = DataParallel_Loss()