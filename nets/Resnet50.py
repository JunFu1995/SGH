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
    """
    Hyper network for learning perceptual rules.

    Args:
        lda_out_channels: local distortion aware module output size.
        hyper_in_channels: input feature channels for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        feature_size: input feature map width/height for hyper network.

    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.

    """
    def __init__(self, ):
        super().__init__()

        self.res = resnet50_backbone(16, 112, pretrained=True, use_scale=0)

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(2048,256),nn.ReLU())
        # initialize
        # for i, m_name in enumerate(self._modules):
        #     if i > 2:
        #         nn.init.kaiming_normal_(self._modules[m_name].weight.data)
        # self.w = nn.Linear(1, 2048)
        # self.fc3 = nn.Linear(2048, 1)

        # self.fc = nn.Sequential(
        #     nn.Linear(2048*2, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        #     )
        self.hyperFc = hyperFc(256)
    def forward(self, img):
        img, z = img 
        x = self.res(img)
        x = self.pool(x)
        x = x.view(x.shape[0], -1) 
        x = self.fc(x)
        x = self.hyperFc(x, z)
        # z = self.w(z)
        # z = z.repeat(img.shape[0]//z.shape[0],1)
        # print(z.shape)
        # x = torch.cat((x, z),dim=1)

        #mean, std = torch.mean(x, dim=1, keepdim=True), torch.std(x, dim=1, keepdim=True) + 1e-8
        # + self.fc1(z)
        #x = self.fc1(z) * (x -mean)/std + self.fc2(z)
        # x = self.fc(x)
        return x



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetBackbone(nn.Module):

    def __init__(self, lda_out_channels, in_chn, block, layers, num_classes=1000, use_scale=False):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.use_scale = use_scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #print(m)
                nn.init.kaiming_normal_(m.weight.data)

        # # initialize
        # nn.init.kaiming_normal_(self.lda1_pool._modules['0'].weight.data)
        # nn.init.kaiming_normal_(self.lda2_pool._modules['0'].weight.data)
        # nn.init.kaiming_normal_(self.lda3_pool._modules['0'].weight.data)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        #lda_1 = self.lda1_fc(self.lda1_pool(x).view(x.size(0), -1))
        x = self.layer2(x)
        #lda_2 = self.lda2_fc(self.lda2_pool(x).view(x.size(0), -1))
        x = self.layer3(x)
        #lda_3 = self.lda3_fc(self.lda3_pool(x).view(x.size(0), -1))
        x = self.layer4(x) #
        #lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))
             
        return x

def resnet50_backbone(lda_out_channels, in_chn, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3], **kwargs)
    print('pre',pretrained)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


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
