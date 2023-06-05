import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
class TargetNet(nn.Module):
    """
    Target network for quality prediction.
    """
    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            TargetFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        self.l4 = nn.Sequential(
            TargetFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            TargetFC(paras['target_fc5w'], paras['target_fc5b']),
        )


    def forward(self, x):
        q = self.l1(x)
        q = self.l2(q)
        q = self.l3(q)
        q = self.l4(q).squeeze()
        return q


class TargetFC(nn.Module):
    """
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):

        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])

class hyperFc(nn.Module):
    def __init__(self, target_in_size):
        super().__init__()
        feature_size = 1 
        self.target_in_size, self.f1, self.f2, self.f3, self.f4 = target_in_size, 128, 64, 32, 16    
        self.hyperInChn = target_in_size 
        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,  padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.w1 = nn.Linear(1, target_in_size)
        # self.w2 = nn.Linear(1, 1)
        self.w3 = nn.Linear(target_in_size, target_in_size)
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

    def forward(self, merge, z):
        s = self.w1(z)
        s = s.repeat(merge.shape[0]//z.shape[0],1)

        hyper_in_feat = self.w3(s).unsqueeze(-1).unsqueeze(-1) #torch.cat((merge, s.repeat(merge.shape[0]//s.shape[0],1)), dim=1)).unsqueeze(-1).unsqueeze(-1) #(self.w3(merge) * gamma + beta).unsqueeze(-1).unsqueeze(-1) # self.w2(torch.cat((self.w3(merge),z.repeat(merge.shape[0]//z.shape[0],1)),dim=1)).unsqueeze(-1).unsqueeze(-1)

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)      


        out = {}
        out['target_in_vec'] = merge.unsqueeze(-1).unsqueeze(-1) 
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        paras = out 
        model_target = TargetNet(paras).cuda()
        for param in model_target.parameters():
            param.requires_grad = False

        # Quality prediction
        pred = model_target(paras['target_in_vec'])
        return pred
