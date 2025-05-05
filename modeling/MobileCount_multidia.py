"""RefineNet-LightWeight. No RCU, only LightWeight-CRP block."""

import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
import torchvision.models as models

model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

# Helpers / wrappers
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes*expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes*expansion)
        self.conv2 = nn.Conv2d(inplanes*expansion, inplanes*expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=inplanes*expansion)
        self.bn2 = nn.BatchNorm2d(inplanes*expansion)
        self.conv3 = nn.Conv2d(inplanes*expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
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


class MobileCount(nn.Module):

    def __init__(self, num_classes=1, pretrained=False):
        self.inplanes = 32
        block = Bottleneck
        layers = [1, 2, 3, 4]
        super(MobileCount, self).__init__()

        # Modification 1: Use low-momentum BatchNorm2d in FIDTM
        # Low-momentum BatchNorm helps improve the stability of training on small datasets.
        self.bn_momentum = 0.1

        # Keep the lightweight nature of MobileCount
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_momentum)  # 使用低动量
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1, expansion=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, expansion=6)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, expansion=6)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, expansion=6)

        # Modification point 2: Introduce the advanced upsampling module in FIDTM
        # Add deconvolution layers to improve upsampling quality
        self.upsample1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample1 = nn.BatchNorm2d(128, momentum=self.bn_momentum)
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample2 = nn.BatchNorm2d(64, momentum=self.bn_momentum)

        # Modification point 3: Improve the feature fusion mechanism and add more cross-resolution information
        # Introducing high-resolution feature fusion similar to FIDTM
        self.fuse_conv = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_fuse = nn.BatchNorm2d(32, momentum=self.bn_momentum)

        # Multi-Column Dilated Convolution Module 保持不变
        self.conv4_3_1 = torch.nn.Conv2d(32, 8, kernel_size=3, padding=2, dilation=2)
        self.conv4_3_2 = torch.nn.Conv2d(32, 8, kernel_size=3, padding=4, dilation=4)
        self.conv4_3_3 = torch.nn.Conv2d(32, 8, kernel_size=3, padding=8, dilation=8)
        self.conv4_3_4 = torch.nn.Conv2d(32, 8, kernel_size=3, padding=12, dilation=12)
        self.conv5 = torch.nn.Conv2d(32, 32, kernel_size=1)

        self.dropout = nn.Dropout(p=0.5)
        self.clf_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, expansion):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        size1 = x.shape[2:]

        # Input convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Multi-Column Dilated Convolution Module
        x1 = self.conv4_3_1(x)
        x1 = F.softmax(x1, dim=1)
        x2 = self.conv4_3_2(x)
        x2 = F.softmax(x2, dim=1)
        x3 = self.conv4_3_3(x)
        x3 = F.softmax(x3, dim=1)
        x4 = self.conv4_3_4(x)
        x4 = F.softmax(x4, dim=1)
        x = torch.cat((x1, x2, x3, x4), 1)

        x = self.conv5(x)
        x = self.maxpool(x)

        # Backbone network
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        # Advanced upsampling and fusion
        l4 = self.upsample1(l4)
        l4 = self.bn_upsample1(l4)
        l4 = F.relu(l4)

        l4 = self.upsample2(l4)
        l4 = self.bn_upsample2(l4)
        l4 = F.relu(l4)

        fused = self.fuse_conv(l4 + l2)  # 融合高分辨率和低分辨率特征
        fused = self.bn_fuse(fused)
        fused = F.relu(fused)

        fused = self.dropout(fused)
        out = self.clf_conv(fused)

        # Output upsampled to original size
        out = F.interpolate(out, size=size1, mode='bilinear')

        return out
