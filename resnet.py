# resnet.py

import torch
import torch.nn as nn
from qil import QConv2d, QActivation
from torchsummary import summary
import os
from torch.nn.parallel import DistributedDataParallel, DataParallel


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, w_bit=32, a_bit=32):
        super(PreActBasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.quant_activation_1 = QActivation(a_bit)
        self.quant_conv1 = QConv2d(in_planes, planes, 3, stride, padding=1, bit=w_bit)

        self.bn2 = nn.BatchNorm2d(planes)
        self.quant_activation_2 = QActivation(a_bit)
        self.quant_conv2 = QConv2d(planes, planes, 3, padding=1, bit=w_bit)

        if stride != 1 or in_planes != planes * PreActBasicBlock.expansion:
            self.shortcut = nn.Sequential(
                QConv2d(in_planes, planes * PreActBasicBlock.expansion, 1, stride, bit=w_bit))

    def forward(self, x):

        """ Pre Activate Residual Block """

        out = self.bn1(x)
        out = self.relu(out)
        out = self.quant_activation_1(out)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.quant_conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.quant_activation_2(out)
        out = self.quant_conv2(out)

        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, w_bit=32, a_bit=32):
        super(ResNet, self).__init__()

        self.num_classes = num_classes
        self.norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.w_bit = w_bit
        self.a_bit = a_bit

        if num_classes == 10:
            self.in_planes = 16
            conv1_param = [3, 1, 1]
            make_layer_param = [16, 32, 64]
            fc_inplain = 64

        elif num_classes == 1000:
            self.in_planes = 64
            conv1_param = [7, 2, 3]
            make_layer_param = [64, 128, 256, 512]
            fc_inplain = 512

        """ first Layer doesn't apply qantization """
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=conv1_param[0], stride=conv1_param[1], padding=conv1_param[2], bias=False)
        self.bn1 = self.norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        """ Average Pooling """
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        """ Max Pooling """
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        """ Quantization """
        self.layer1 = self._make_layer(block, make_layer_param[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, make_layer_param[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, make_layer_param[2], layers[2], stride=2)
        if num_classes == 1000:
            self.layer4 = self._make_layer(block, make_layer_param[3], layers[3], stride=2)

        """ last bn and relu for pre-activation """
        self.last_bn = self.norm_layer(fc_inplain * block.expansion)
        self.last_relu = nn.ReLU(inplace=True)


        """ Final Layer doesn't apply quantization """
        self.fc = nn.Linear(fc_inplain * block.expansion, num_classes)


    def _make_layer(self, block, planes, layer, stride=None):

        layers = []

        layers.append(block(self.in_planes, planes, stride, w_bit=self.w_bit, a_bit=self.a_bit))

        self.in_planes = planes * block.expansion
        for num_block in range(1, layer):
            layers.append(block(self.in_planes, planes, w_bit=self.w_bit, a_bit=self.a_bit))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # FIXME: maxpooling 버그인가...?
        # if self.num_classes == 1000:
        x = self.maxpooling(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # if self.num_classes == 1000:
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet20(pretrained_path, num_classes, **kwargs):
    resnet = ResNet(PreActBasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)

    if pretrained_path is not None:
        if not os.path.exists(pretrained_path):
            print(f'CIFAR10 FP32 학습 시작')
            return resnet
        print(f'load: {kwargs} | {pretrained_path}')

        checkpoint = torch.load(f'{pretrained_path}')
        resnet.load_state_dict(checkpoint['state_dict'], strict=True)
    return resnet


def resnet18(pretrained_path, num_classes, **kwargs):
    resnet = ResNet(PreActBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

    checkpoint = torch.load(f'{pretrained_path}')
    if isinstance(resnet, (DataParallel, DistributedDataParallel)):
        resnet.module.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        resnet.load_state_dict(checkpoint['state_dict'], strict=False)
    print('Done checkpoint load')

    return resnet


if __name__ == '__main__':
    model = resnet20()
    print(model)
    summary(model, (3, 32, 32), depth=3)