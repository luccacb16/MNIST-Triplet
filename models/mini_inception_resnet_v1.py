import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class InceptionResnetBlock(nn.Module):
    def __init__(self, scale, block_type, in_channels, out_channels):
        super(InceptionResnetBlock, self).__init__()
        if block_type == 'block35':
            self.branch0 = BasicConv2d(in_channels, out_channels, kernel_size=1)
            self.branch1 = nn.Sequential(
                BasicConv2d(in_channels, out_channels, kernel_size=1),
                BasicConv2d(out_channels, out_channels, kernel_size=5, padding=2)
            )
            self.branch2 = nn.Sequential(
                BasicConv2d(in_channels, out_channels, kernel_size=1),
                BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1),
                BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
            self.conv2d = nn.Conv2d(3 * out_channels, in_channels, kernel_size=1)
            self.scale = scale
            self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale
        return self.relu(residual + out)

class MiniInceptionResNetV1(nn.Module):
    def __init__(self, emb_size=128):
        super(MiniInceptionResNetV1, self).__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = InceptionResnetBlock(scale=0.17, block_type='block35', in_channels=192, out_channels=32)
        
        self.avgpool_1a = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.6)
        self.last_linear = nn.Linear(192, emb_size)
        self.last_bn = nn.BatchNorm1d(emb_size, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        
        x = self.avgpool_1a(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.last_linear(x)
        x = self.last_bn(x)
        x = self.relu(x)
        
        # L2 Normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x