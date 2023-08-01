import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18
import numpy as np
import math
import matplotlib.pyplot as plt

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
        
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)
        
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num):
        super(ClassBlock, self).__init__()
        
        self.weight = torch.nn.Parameter(torch.randn(class_num, input_dim), requires_grad=True)
        nn.init.normal_(self.weight, 0, 0.001)

    def forward(self, x):
        x = F.linear(x, self.weight)
        return x, self.weight


class IDA_classifier(nn.Module):
    def __init__(self, num_part, class_num):
        super(IDA_classifier, self).__init__()
        input_dim = 2048
        self.l2norm = Normalize(2) 
        self.part = num_part
        for i in range(num_part):
            name = 'classifierD_' + str(i)
            setattr(self, name, ClassBlock(input_dim, class_num))

    def forward(self, x):
        proxy = {}
        out = {}
        start_point = len(out)
        for i in range(self.part):
            name = 'classifierD_' + str(i)
            cls_part = getattr(self, name)
            out[i + start_point], proxy[i + start_point] = cls_part(x[i])
            proxy[i + start_point] = self.l2norm(proxy[i + start_point])
        return out, proxy