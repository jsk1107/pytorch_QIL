# resnet.py

import torch
import torch.nn as nn
from qil import QConv2d, QActivation
# from test import Quant_Conv2d, Quant_Activation
from torchsummary import summary
import os


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, w_bit=32, a_bit=32):
        super(BasicBlock, self).__init__()

        self.quant_conv1 = QConv2d(in_planes, planes, 3, stride, padding=1, bit=w_bit)
        self.bn1 = nn.BatchNorm2d(planes)
        self.quant_activation_1 = QActivation(a_bit)
        self.quant_conv2 = QConv2d(planes, planes, 3, padding=1, bit=w_bit)
        self.bn2 = nn.BatchNorm2d(planes)
        self.quant_activation_2 = QActivation(a_bit)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.quant_conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.quant_activation_1(out)

        out = self.quant_conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.relu(out)
        out = self.quant_activation_2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, w_bit=32, a_bit=32):
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

        """ Final Layer doesn't apply quantization """
        self.fc = nn.Linear(fc_inplain * block.expansion, num_classes)


        # TODO
        """ Init Weight """

    def _make_layer(self, block, planes, layer, stride=None):

        downsample = None
        layers = []

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                QConv2d(self.in_planes, planes, 1, stride, bit=self.w_bit),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers.append(block(self.in_planes, planes, stride, downsample, w_bit=self.w_bit, a_bit=self.a_bit))

        self.in_planes = planes * block.expansion
        for num_block in range(1, layer):
            layers.append(block(self.in_planes, planes, w_bit=self.w_bit, a_bit=self.a_bit))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpooling(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        """ 이미지넷 클래스 개수: 1000 """
        """ CIFAR10 클래스 갯수: 10 """
        if self.num_classes == 1000:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet20(pretrained_path=None, **kwargs):
    resnet = ResNet(BasicBlock, [3, 3, 3], num_classes=10, **kwargs)

    if pretrained_path is not None:
        if not os.path.exists(pretrained_path):
            print(f'CIFAR10 FP32 학습 시작')
            return resnet
        print(f'load: {kwargs} | {pretrained_path}')

        checkpoint = torch.load(f'{pretrained_path}')
        resnet.load_state_dict(checkpoint['state_dict'], strict=True)
    return resnet


def resnet18(pretrained_path=None, **kwargs):
    resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000, **kwargs)

    if pretrained_path is not None:
        if not os.path.exists(pretrained_path):
            print(f'ImageNet FP32 학습 시작')
            return resnet
        print(f'load: {kwargs} | {pretrained_path}')

        checkpoint = torch.load(f'{pretrained_path}')
        resnet.load_state_dict(checkpoint['state_dict'], strict=True)
    return resnet


if __name__ == '__main__':
    model = resnet20()
    print(model)
    summary(model, (3, 32, 32), depth=3)