import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from .modulate import * 

def conv7x7(in_channels, out_channels, stride=1, padding=3, dilation=1):
    ''' 7x7 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=padding, dilation=dilation, bias=False)

class CBAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio, dilation=1):
        super(CBAM, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP.
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=self.hid_channel),
            nn.ReLU(),
            nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = conv7x7(2, 1, stride=1, dilation=self.dilation)

    def forward(self, x):
        ''' Channel attention '''
        avgOut = self.globalAvgPool(x)
        avgOut = avgOut.view(avgOut.size(0), -1)
        avgOut = self.mlp(avgOut)

        maxOut = self.globalMaxPool(x)
        maxOut = maxOut.view(maxOut.size(0), -1)
        maxOut = self.mlp(maxOut)
        # sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))
        Mc = self.sigmoid(avgOut + maxOut)
        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)
        Mf1 = Mc * x

        ''' Spatial attention. '''
        # sigmoid(conv7x7( [AvgPool(F); MaxPool(F)]))
        maxOut = torch.max(Mf1, 1)[0].unsqueeze(1)
        avgOut = torch.mean(Mf1, 1).unsqueeze(1)
        Ms = torch.cat((maxOut, avgOut), dim=1)

        Ms = self.conv1(Ms)
        Ms = self.sigmoid(Ms)
        Ms = Ms.view(Ms.size(0), 1, Ms.size(2), Ms.size(3))
        Mf2 = Ms * Mf1
        return Mf2


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CBAM(nn.Module):
#     def __init__(self, in_planes, ratio, kernel_size):
#         super().__init__()

#         self.ca = ChannelAttention(in_planes, ratio)
#         self.sa = SpatialAttention(kernel_size)

#     def forward(self, x):
#         out = x 
#         out = self.ca(out) * out
#         out = self.sa(out) * out
#         return out  

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
    def __init__(self):
        super().__init__()
        self.deepBranch = nn.Sequential(
                    nn.Conv2d(3, 50, 7),
                   nn.Conv2d(50, 50, 3),                     
                   nn.Conv2d(50, 50, 3),
                   CBAM(50, 4),
            )
        self.lightBranch = nn.Sequential(
                    nn.Conv2d(3, 50, 7)
            )
        self.pool = Pool()
        self.AG = nn.Conv2d(100,100,1)
        # self.w = nn.Linear(1,200)

        # self.fc = nn.Sequential(
        #     nn.Linear(200*2, 800),
        #     nn.ReLU(),
        #     #nn.Dropout(0.5),
        #     nn.Linear(800, 800),
        #     nn.ReLU(),
        #     nn.Linear(800, 1)
        #     )
        self.hyperFc = hyperFc(200)
    def forward(self, x):
        x, z = x 
        arb = self.deepBranch(x)
        srb = self.lightBranch(x)
        arb = self.AG(self.pool(arb))
        srb = self.AG(self.pool(srb))

        merge = torch.cat((arb, srb), dim=1)
        merge = merge.view(arb.shape[0], -1)
        q = self.hyperFc(merge, z)
        # z = self.w(z)
        # z = z.repeat(x.shape[0]//z.shape[0],1)
        # merge = torch.cat((merge,z),dim=1)
        # q = self.fc(merge)
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
