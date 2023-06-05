import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from .modulate import * 
class ResidualBlock_CRes(nn.Module):
    '''Residual block with controllable residual connections
    ---Conv-ReLU-Conv-x-+-
     |________________|
    '''

    def __init__(self, nf=64, cond_dim=2):
        super(ResidualBlock_CRes, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

        self.local_scale = nn.Linear(cond_dim, nf, bias=True)

        # initialization

    def forward(self, x):
        identity = x[0]
        cond = x[1]

        local_scale = self.local_scale(cond)
        out = self.conv1(identity)
        out = self.conv2(self.act(out))
        return (identity + out * local_scale.unsqueeze(-1).unsqueeze(-1),cond)
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
        self.conv2 = nn.Sequential(ResidualBlock_CRes(50,1),ResidualBlock_CRes(50,1))
        self.pool = Pool()
        self.fc = nn.Sequential(
            nn.Linear(100, 800),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 1)
            )
        # self.fc1 = nn.Linear(1, 100)
        # self.fc2 = nn.Linear(1, 1)
        # 
        #self.hyperFc = hyperFc(100)
    def forward(self, x):
        x, z = x 
        x  = self.conv1(x)  #
        x, _ = self.conv2((x,z))
        x = self.pool(x)
        x = x.view(x.shape[0],-1)
        # q = self.hyperFc(x, z)
        #print(z.shape)
        # z = z.repeat(x.shape[0]//z.shape[0],1)
        # x = torch.cat((x,self.fc1(z)),dim=1)
        #mean, std = torch.mean(x, dim=1, keepdim=True), torch.std(x, dim=1, keepdim=True) + 1e-8
        # + self.fc1(z)
        #x = self.fc1(z) * (x -mean)/std + self.fc2(z)
        q  = self.fc(x)# * self.fc1(z) + self.fc2(z)
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
