import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from .modulate import * 
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.texture_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ELU(),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),
            )

        self.texture_fc = nn.Sequential(
            nn.Linear(16*64*49, 128),
            nn.ELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            )

        self.structure_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ELU(),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),
            )

        self.structure_fc = nn.Sequential(
            nn.Linear(16*64*49, 128),
            nn.ELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            )

        # self.w = nn.Linear(1, 256)
        # self.fc = nn.Linear(512, 1)

        self.hyperFc = hyperFc(256)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         #print(m)
        #         nn.init.kaiming_normal_(m.weight.data)


    def forward(self, img):
        img, z = img 
        texture, structure = torch.split(img,3,1) 
        #print(texture.shape)
        texture = self.texture_conv(texture)
        #print(texture.shape)
        texture = texture.view(texture.shape[0],-1)
        texture = self.texture_fc(texture)

        structure = self.texture_conv(structure)
        structure = structure.view(structure.shape[0],-1)
        structure = self.texture_fc(structure)  

        # z = self.w(z)
        # z = z.repeat(img.shape[0]//z.shape[0],1)
        merge = torch.cat((texture, structure), dim=1)

        pred = self.hyperFc(merge, z)
        return pred

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
