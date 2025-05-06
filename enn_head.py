import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(num_classes)
            self.linear = nn.Linear(in_features, num_classes, bias=False)
        else:
            self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        if self.use_bn:
            logits = self.bn(logits)
        evidence = F.softplus(logits)
        return evidence
