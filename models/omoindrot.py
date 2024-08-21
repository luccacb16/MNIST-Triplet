import torch
import torch.nn as nn
import torch.nn.functional as F

# Modelo obtido de: https://github.com/omoindrot/tensorflow-triplet-loss

class Omoindrot(nn.Module):
    def __init__(self, emb_size=64):
        super(Omoindrot, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 32 * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32 * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.Linear(32 * 2 * 7 * 7, emb_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        x = F.normalize(x, p=2, dim=1)

        return x