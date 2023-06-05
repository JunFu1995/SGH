import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from .modulate import * 

class Pool(nn.Module):
    def __init__(self):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = torch.cat((avg_out, max_out), dim=1)
        return out

class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 50, 7) 
        self.pool = Pool()
        # self.fc = nn.Sequential(
        #     nn.Linear(200, 800),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(800, 800),
        #     nn.ReLU(),
        #     nn.Linear(800, 1)
        #     )
        # self.fc1 = nn.Linear(1, 100)
        # self.fc2 = nn.Linear(1, 1)
        # 
        self.hyperFc = hyperFc(100)
    def forward(self, x):
        x, z = x 
        x  = self.conv1(x)  #
        x = self.pool(x)
        x = x.view(x.shape[0],-1)
        q = self.hyperFc(x, z)
        #print(z.shape)
        # z = z.repeat(x.shape[0]//z.shape[0],1)
        # x = torch.cat((x,self.fc1(z)),dim=1)
        #mean, std = torch.mean(x, dim=1, keepdim=True), torch.std(x, dim=1, keepdim=True) + 1e-8
        # + self.fc1(z)
        #x = self.fc1(z) * (x -mean)/std + self.fc2(z)
        #q  = self.fc(x)# * self.fc1(z) + self.fc2(z)
        return q


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
