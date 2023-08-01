import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
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

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)



class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num):
        super(ClassBlock, self).__init__()
        
        self.bottleneck = nn.BatchNorm1d(input_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat = self.bottleneck(x)
        out = self.classifier(feat)
        return feat, out


class classifier(nn.Module):
    def __init__(self, num_part, class_num):
        super(classifier, self).__init__()
        input_dim = 2048
        self.part = num_part
        self.l2norm = Normalize(2)
        for i in range(num_part):
            name = 'classifier_' + str(i)
            setattr(self, name, ClassBlock(input_dim, class_num))

    def forward(self, x, feat_all, out_all):
        start_point = len(feat_all)
        for i in range(self.part):
            name = 'classifier_' + str(i)
            cls_part = getattr(self, name)
            feat_all[i + start_point], out_all[i + start_point] = cls_part(torch.squeeze(x[:, :, i]))

        return feat_all, out_all
        
        
class embed_net(nn.Module):
    def __init__(self,  class_num, part, arch='resnet50'):
        super(embed_net, self).__init__()
        
        self.part = part
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.classifier = classifier(part, class_num)
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block
        x = self.base_resnet(x)
        x = self.avgpool(x)
        feat = {}
        out = {}
        feat, out = self.classifier(x, feat, out)
        if self.training:
            return feat, out
        else:
            for i in range(self.part):
                if i == 0:
                    featf = self.l2norm(feat[i])
                else:
                    featf = torch.cat((featf, self.l2norm(feat[i])), 1)
            return self.l2norm(featf)