import torch
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(3, 128, 3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(128, 64, 3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(64, 32, 3, stride=2, padding=1)
#         self.mlp1 = nn.Linear(2, 1024)
#         self.mlp2 = nn.Linear(1024, 2)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.mlp1(x)
#         x = self.mlp2(x)
#         return x
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.xavier_uniform(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant(m.weight, 1)
#                 init.constant(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal(m.weight, std=1e-3)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.mlp1 = nn.Linear(8192, 512)
        self.mlp2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((x.shape[0], -1))
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)
