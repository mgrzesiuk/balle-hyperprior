from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class BalleHyperpriorEncoder(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.conv1 = nn.Conv2d(N, N, 3, 1)
        self.conv2 = nn.Conv2d(N, N, 5, 2)
        self.conv3 = nn.Conv2d(N, N, 5, 2)

    def forward(self, x: Tensor):
        x = x.abs()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return self.conv3(x)