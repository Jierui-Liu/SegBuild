# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models

class FullModel(nn.Module):
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, targets, *inputs):
    outputs = self.model(*inputs)
    loss = self.loss(outputs, targets)
    return torch.unsqueeze(loss,0),outputs
    

def DataParallel_withLoss(model,loss,**kwargs):
    model=FullModel(model, loss)
    if 'device_ids' in kwargs.keys():
        device_ids=kwargs['device_ids']
    else:
        device_ids=None
    if 'output_device' in kwargs.keys():
        output_device=kwargs['output_device']
    else:
        output_device=None
    if 'cuda' in kwargs.keys():
        cudaID=kwargs['cuda'] 
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda(cudaID)
    else:
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda()
    return model
class toy(nn.Module):
    def __init__(self):
        super(toy, self).__init__()
        self.conv2d = torch.nn.Conv2d(1,3,1)
    def forward(self,x):
        return self.conv2d(x)


if __name__ == "__main__":
    model = models.segmentation.deeplabv3_resnet50(num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(),lr=1)
    loss = torch.nn.L1Loss()
    model = DataParallel_withLoss(model,loss)
    for i in range(10000):
        gt = torch.zeros(1,1,224,224)
        input = torch.rand(1,3,224,224)
        loss,_ = model(gt,input)
        loss = loss.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()