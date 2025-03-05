import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialClassificationHead(nn.Module):
    """
    A simple linear + softplus head that produces nonnegative evidence for each class.
    """
    def __init__(self, in_features, num_classes, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(in_features)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
          x: (B, in_features) - feature vectors from the backbone or pooled representation
        Returns:
          evidence: (B, num_classes) - nonnegative evidence for each class
        """
        if self.use_bn:
            x = self.bn(x)
        logits = self.linear(x)
        evidence = F.softplus(logits)
        return evidence
