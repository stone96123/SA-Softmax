import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
        
class proxyMSE(nn.Module):
    def __init__(self):
        super(proxyMSE, self).__init__()
       
        self.MSE = nn.MSELoss(reduction='sum')
        self.l2norm = Normalize(2)
        
    def forward(self, feat, proxy, one_hot):
        proxy = torch.matmul(one_hot, proxy)
        dist = torch.mul(self.l2norm(feat), self.l2norm(proxy))
        dist = torch.sum(dist, dim=1)
        loss = self.MSE(dist, torch.ones(dist.shape).cuda())
        return loss / feat.size()[0]