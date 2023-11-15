#!/usr/bin/env python3.8

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.bn1(y1)
        y2 = self.conv2(x)
        y2 = self.bn2(y2)
        return torch.cat([y1, y2], 1)

module = Net()

symbolic_traced = symbolic_trace(module)

print(symbolic_traced.graph)

