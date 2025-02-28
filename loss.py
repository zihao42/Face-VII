import torch
import torch.nn as nn


def variance_aware_loss_from_batch(z, logits, labels, lambda_reg=0.05, lambda_cls=1.0):
    """
    参数：
      z: Tensor, shape (B, 768) - 当前批次的特征向量
      logits: Tensor, shape (B, K) - 闭集分类输出，用于计算交叉熵损失
      labels: Tensor, shape (B,) - 真实标签，取值范围为 0 ~ (K-1)
      lambda_reg: 正则化损失的平衡系数
      lambda_cls: 分类损失的平衡系数
    返回：
      loss: 总损失
      Ldist, Lreg, Lcls: 各部分损失（便于调试）
    """
    eps = 1e-6  # 防止除 0
    unique_labels = labels.unique()

    # 用字典存储当前批次中每个类别的统计量
    class_stats = {}
    for cl in unique_labels:
        cl_mask = (labels == cl)
        z_cl = z[cl_mask]  # 当前类别的所有特征，形状 (N_cl, 768)
        mu_cl = z_cl.mean(dim=0)  # 均值，形状 (768,)
        var_cl = z_cl.var(dim=0, unbiased=False) + eps  # 方差，形状 (768,)
        class_stats[int(cl.item())] = (mu_cl, var_cl)

    # 对每个样本，查找其所属类别的统计量
    mu_batch = torch.stack([class_stats[int(lbl.item())][0] for lbl in labels], dim=0)  # (B, 768)
    var_batch = torch.stack([class_stats[int(lbl.item())][1] for lbl in labels], dim=0)  # (B, 768)

    # 计算 Ldist: 对角协方差下的马氏距离（忽略常数项）
    diff = z - mu_batch
    Ldist = 0.5 * torch.sum(diff * diff / var_batch, dim=1).mean()

    # 计算 Lreg: 每个样本对数行列式（对角矩阵时等于各维度 log(var) 的和）
    Lreg = 0.5 * torch.sum(torch.log(var_batch), dim=1).mean()

    # 计算闭集分类损失 Lcls (交叉熵)
    ce_loss = nn.CrossEntropyLoss()(logits, labels)

    # 联合总损失(Ldist是马氏距离损失，Lreg是方差损失，ce_loss是分类损失（交叉熵））
    loss = Ldist + lambda_reg * Lreg + lambda_cls * ce_loss
    return loss, Ldist, Lreg, ce_loss

def scheduled_variance_aware_loss(z, logits, labels, current_epoch, total_epochs, lambda_reg=0.05, lambda_cls=1.0):
    """
    Computes the scheduled variance-aware loss for a given batch.
    
    Parameters:
      z: Tensor of shape (B, 768) representing the feature vectors for the current batch.
      logits: Tensor of shape (B, K) representing the classification outputs for the closed-set task.
      labels: Tensor of shape (B,) containing the ground-truth labels (values in 0 to K-1).
      current_epoch: Current training epoch.
      total_epochs: Total number of training epochs.
      lambda_reg: Weighting coefficient for the regularization loss (Lreg).
      lambda_cls: Weighting coefficient for the classification loss (Lcls).
      
    Returns:
      loss: The overall loss for the batch.
      Ldist: The Mahalanobis distance-based loss component.
      Lreg: The regularization loss component.
      ce_loss: The cross-entropy (classification) loss.
      alpha: The scheduling factor used to weight Ldist and Lreg.
    """
    eps = 1e-6  # Small constant to avoid division by zero
    B, M = z.shape
    unique_labels = labels.unique()
    
    # Compute the mean and variance for each class in the current batch.
    # Statistics are computed per class based on the feature vectors.
    class_stats = {}
    for cl in unique_labels:
        cl_mask = (labels == cl)
        z_cl = z[cl_mask]          # Feature vectors for class 'cl', shape (N_cl, 768)
        mu_cl = z_cl.mean(dim=0)     # Mean for class 'cl', shape (768,)
        var_cl = z_cl.var(dim=0, unbiased=False) + eps  # Variance for class 'cl', shape (768,)
        class_stats[int(cl.item())] = (mu_cl, var_cl)
    
    # Gather the statistics for each sample in the batch.
    mu_batch = torch.stack([class_stats[int(lbl.item())][0] for lbl in labels], dim=0)   # Shape (B, 768)
    var_batch = torch.stack([class_stats[int(lbl.item())][1] for lbl in labels], dim=0)  # Shape (B, 768)
    
    # Compute Ldist: Mahalanobis distance under the assumption of a diagonal covariance matrix.
    diff = z - mu_batch
    Ldist = 0.5 * torch.sum(diff * diff / var_batch, dim=1).mean()
    
    # Compute Lreg: Regularization term based on the log-determinant of the diagonal covariance matrix.
    # For a diagonal matrix, log|Σ| = sum(log(var)).
    Lreg = 0.5 * torch.sum(torch.log(var_batch), dim=1).mean()
    
    # Compute the classification loss Lcls using cross-entropy.
    ce_loss = nn.CrossEntropyLoss()(logits, labels)
    
    # Compute the scheduling factor alpha.
    # For the first half of the epochs, alpha=0 (only classification loss is used).
    # For the second half, alpha increases linearly from 0 to 1.
    half_epoch = total_epochs / 2
    if current_epoch < half_epoch:
        alpha = 0.0
    else:
        alpha = (current_epoch - half_epoch) / half_epoch
        alpha = min(alpha, 1.0)
    
    # Final loss: Always include the classification loss; gradually add Ldist and Lreg after half of the training.
    loss = lambda_cls * ce_loss + alpha * (Ldist + lambda_reg * Lreg)
    
    return loss, Ldist, Lreg, ce_loss, alpha
