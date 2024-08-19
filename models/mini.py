import torch.nn as nn
import torch.nn.functional as F

class L2NormSquared(nn.Module):
    def __init__(self):
        super(L2NormSquared, self).__init__()

    def forward(self, x):
        return x.pow(2).sum(1)

class MiniNet(nn.Module):
    def __init__(self, emb_size=64):
        super(MiniNet, self).__init__()
        self.l2 = L2NormSquared()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, emb_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)

        x = self.l2(x)
        
        return x