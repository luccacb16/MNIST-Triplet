import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNetwork(nn.Module):
    def __init__(self, num_channels, emb_size):
        super(TripletNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.Linear(num_channels * 2 * 7 * 7, emb_size)

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

        return x