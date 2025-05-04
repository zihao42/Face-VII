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
        self.ln_logits = nn.LayerNorm(num_classes)
        if use_bn:
            self.bn = nn.BatchNorm1d(num_classes)
            self.linear = nn.Linear(in_features, num_classes, bias=False)
        else:
            self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
          x: (B, in_features) - feature vectors from the backbone or pooled representation
        Returns:
          evidence: (B, num_classes) - nonnegative evidence for each class
        """
        logits = self.linear(x)
        logits = self.ln_logits(logits)
        if self.use_bn:
            logits = self.bn(logits)
        evidence = F.softplus(logits)
        return evidence
