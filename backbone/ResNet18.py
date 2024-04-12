# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.functional import relu, avg_pool2d
from typing import List, Tuple
from itertools import repeat

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """

    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])  # 默认沿着中间所有的shape
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)

# class CosineLinear(nn.Module):
#     def __init__(self, in_features, out_features, sigma=True):
#         super(CosineLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         if sigma:
#             self.sigma = Parameter(torch.Tensor(1))
#         else:
#             self.register_parameter('sigma', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.sigma is not None:
#             self.sigma.data.fill_(1)  # for initializaiton of sigma
#
#     def forward(self, input):
#         out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
#         if self.sigma is not None:
#             out = self.sigma * out
#         return out


class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
        self.dout = SpatialDropout(drop=0.5)

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )

        self.classifier = self.linear  # self.linear  #

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        h0 = relu(self.bn1(self.conv1(x)))
        h0 = self.maxpool(h0)
        h1 = self.layer1(h0)  # 64, 32, 32
        h1 = self.dout(h1)
        h2 = self.layer2(h1)  # 128, 16, 16
        h2 = self.dout(h2)
        h3 = self.layer3(h2)  # 256, 8, 8
        h3 = self.dout(h3)
        h4 = self.layer4(h3)  # 512, 4, 4
        h4 = self.dout(h4)

        out = avg_pool2d(h4, h4.shape[2])  # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        # out = F.normalize(out, dim=0, p=2)
        out = self.linear(out)
        return out

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        feat = self._features(x)
        out = avg_pool2d(feat, feat.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return feat, out

    def extract_features(self, x: torch.Tensor) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = relu(self.bn1(self.conv1(x)))
        feat1 = self.layer1(out)  # 64, 32, 32
        feat2 = self.layer2(feat1)  # 128, 16, 16
        feat3 = self.layer3(feat2)  # 256, 8, 8
        feat4 = self.layer4(feat3)  # 512, 4, 4
        out = avg_pool2d(feat4, feat4.shape[2])  # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)

        return (feat1, feat2, feat3, feat4), out

    def get_features_only(self, x: torch.Tensor, feat_level: int) -> torch.Tensor:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """

        feat = relu(self.bn1(self.conv1(x)))

        if feat_level > 0:
            feat = self.layer1(feat)  # 64, 32, 32
        if feat_level > 1:
            feat = self.layer2(feat)  # 128, 16, 16
        if feat_level > 2:
            feat = self.layer3(feat)  # 256, 8, 8
        if feat_level > 3:
            feat = self.layer4(feat)  # 512, 4, 4
        return feat

    def predict_from_features(self, feats: torch.Tensor, feat_level: int) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param feats: input tensor (batch_size, *input_shape)
        :param feat_level: resnet block
        :return: output tensor (??)
        """

        out = feats

        if feat_level < 1:
            out = self.layer1(out)  # 64, 32, 32
        if feat_level < 2:
            out = self.layer2(out)  # 128, 16, 16
        if feat_level < 3:
            out = self.layer3(out)  # 256, 8, 8
        if feat_level < 4:
            out = self.layer4(out)  # 512, 4, 4

        out = avg_pool2d(out, out.shape[2])  # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)

        return out

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                                               torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def resnet18(nclasses: int, nf: int = 64) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)
