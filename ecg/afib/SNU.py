# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - TorchVision

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNU(nn.Module):
    """ ECG_model """
    def __init__(self, cfg, num_channels, num_classes):
        super(SNU, self).__init__()
        N = cfg.num_blocks // 4
        k = cfg.width_factor
        nGroups = [16 * k, 16 * k, 32 * k, 64 * k, 128 * k]
        self.in_channels = nGroups[0]

        self.conv1 = conv_kx1(num_channels, nGroups[0], cfg.kernel_size, stride=1)
        self.layer1 = self._make_layer(cfg, nGroups[1], N, stride=1,   first_block=True)
        self.layer2 = self._make_layer(cfg, nGroups[2], N, cfg.stride, first_block=False)
        self.layer3 = self._make_layer(cfg, nGroups[3], N, cfg.stride, first_block=False)
        self.layer4 = self._make_layer(cfg, nGroups[4], N, cfg.stride, first_block=False)
        self.bn1 = nn.BatchNorm1d(self.layer4[-1].out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.dropout_rate if cfg.dropout_rate is not None else 0)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(self.in_channels + 5, num_classes)

    def _make_layer(self, cfg, out_channels, num_blocks, stride, first_block):
        layers = []
        for b in range(num_blocks):
            if b == 0: layer = WRN_Block(cfg, self.in_channels, out_channels, stride,   first_block)
            else:      layer = WRN_Block(cfg, self.in_channels, out_channels, stride=1, first_block=False)
            layers.append(layer)
            self.in_channels = layer.out_channels

        return nn.Sequential(*layers)

    def forward(self, x, features, flags=None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.relu(self.bn1(out))
        out = self.dropout(out)

        if flags is not None:
            outputs, outputs_sample = [], []
            for i in range(len(x)):
                outputs_sample.append(out[i:i+1])
                if flags[i]:
                    outputs.append(self.maxpool(torch.cat(outputs_sample, dim=2)))
                    outputs_sample = []
            outputs = torch.cat(outputs, dim=0)
            outputs = outputs.view(len(outputs), -1)
        else:
            outputs = self.maxpool(out)
            outputs = outputs.view(len(outputs), -1)

        outputs = torch.cat([outputs, features], dim=1)
        outputs = self.linear(outputs)

        return outputs


def conv_kx1(in_channels, out_channels, kernel_size, stride=1):
    """ kx1 convolution with padding """
    layers = []
    padding = kernel_size - stride
    padding_left = padding // 2
    padding_right = padding - padding_left
    layers.append(nn.ConstantPad1d((padding_left, padding_right), 0))
    layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=False))
    return nn.Sequential(*layers)


def conv_1x1(in_channels, out_channels):
    """ 1x1 convolution """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)


class WRN_Block(nn.Module):
    """
    WRN Block
    -- BN-ReLU-Conv_kx1 - BN-ReLU-Conv_kx1
    -- (GlobalAvgPool - Conv_1x1-ReLU - Conv_1x1-Sigmoid)
    -- MaxPool-Conv_1x1
    """
    def __init__(self, cfg, in_channels, out_channels, stride, first_block):
        super(WRN_Block, self).__init__()
        self.out_channels = out_channels

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.dropout_rate)
        self.conv1 = conv_kx1(in_channels,  out_channels, cfg.kernel_size, stride)
        self.conv2 = conv_kx1(out_channels, out_channels, cfg.kernel_size, stride=1)
        self.first_block = first_block

        if cfg.se:
            self.se = True
            se_reduction =16
            se_channels = out_channels // se_reduction
            self.sigmoid = nn.Sigmoid()
            self.se_avgpool = nn.AdaptiveAvgPool1d(1)
            self.se_conv1 = conv_1x1(out_channels, se_channels)
            self.se_conv2 = conv_1x1(se_channels, out_channels)
        else:
            self.se = False

        shortcut = []
        if stride != 1:
            shortcut.append(nn.MaxPool1d(stride))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        if self.first_block:
            x = self.relu(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.relu(self.bn1(x))
            out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)

        if self.se:
            se = self.se_avgpool(out)
            se = self.relu(self.se_conv1(se))
            se = self.sigmoid(self.se_conv2(se))
            out = out * se

        x = self.shortcut(x)
        out_c, x_c = out.shape[1], x.shape[1]
        if out_c == x_c: out += x
        else:            out += F.pad(x, (0, 0, 0, out_c - x_c))

        return out

