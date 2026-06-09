
import torch.nn as nn
import itertools
class DCF(nn.Module):
    def __init__(self, channels, rd_ratio=0.25):
        super().__init__()
        self.channels = channels
        reduced_ch = max(1, int(channels * rd_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_ch, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_ch, channels, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, h, w = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)
