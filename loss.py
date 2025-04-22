import torch
import torch.nn as nn
import torch.nn.functional as F

def variance_aware_loss_from_batch(z, logits, labels, lambda_reg=0.02, lambda_cls=1.0):
    eps = 1e-6
    min_var = 1e-1  # 防止方差过小
    unique_labels = labels.unique()

    # 计算每个类别的均值和方差，并对方差下界截断
    class_stats = {}
    for cl in unique_labels:
        cl_mask = (labels == cl)
        z_cl = z[cl_mask]
        mu_cl = z_cl.mean(dim=0)
        var_cl = torch.clamp(z_cl.var(dim=0, unbiased=False) + eps, min=min_var)
        class_stats[int(cl.item())] = (mu_cl, var_cl)
    
    # 为每个样本收集所属类别的统计量
    mu_batch = torch.stack([class_stats[int(lbl.item())][0] for lbl in labels], dim=0)
    var_batch = torch.stack([class_stats[int(lbl.item())][1] for lbl in labels], dim=0)
    
    # 计算马氏距离损失
    diff = z - mu_batch
    Ldist = 0.5 * torch.sum(diff * diff / var_batch, dim=1).mean()
    
    # 新的正则化项：当 var < 1 时，惩罚 -log(var)；当 var >= 1 时，无惩罚
    Lreg = torch.mean(torch.relu(-torch.log(var_batch)))
    
    # 交叉熵分类损失
    ce_loss = nn.CrossEntropyLoss()(logits, labels)
    
    # 总损失
    loss = Ldist + lambda_reg * Lreg + lambda_cls * ce_loss
    return loss, Ldist, Lreg, ce_loss

def scheduled_variance_aware_loss(
    z, logits, labels,
    current_epoch, total_epochs,
    lambda_reg=0.02, lambda_cls=1.0,
    ramp_up_frac: float = 0.67  # 从多少比例的 epoch 开始注入
):
    eps = 1e-6
    min_var = 1e-1
    # 1. 计算 class-wise μ, var（和原版一致）
    unique = labels.unique()
    class_stats = {}
    for cl in unique:
        mask = labels == cl
        z_cl = z[mask]
        mu     = z_cl.mean(0)
        var    = torch.clamp(z_cl.var(0, unbiased=False) + eps, min=min_var)
        class_stats[int(cl)] = (mu, var)
    mu_batch  = torch.stack([class_stats[int(lbl)][0] for lbl in labels])
    var_batch = torch.stack([class_stats[int(lbl)][1] for lbl in labels])
    # 2. Ldist + Lreg
    diff  = z - mu_batch
    Ldist = 0.5 * (diff*diff / var_batch).sum(1).mean()
    Lreg  = torch.mean(torch.relu(-torch.log(var_batch)))
    # 3. CE loss
    ce = nn.CrossEntropyLoss()(logits, labels)

    # 4. 计算 α
    start_epoch = int(total_epochs * ramp_up_frac)
    if current_epoch < start_epoch:
        alpha = 0.0
    else:
        # linear progress in [0,1]
        length = max(1, total_epochs - start_epoch)
        progress = (current_epoch - start_epoch) / length
        # 平滑：linear²  或者直接 linear，都可以
        alpha = min(progress**2, 1.0)

    # 5. 最终 loss
    loss = lambda_cls * ce + alpha * (Ldist + lambda_reg * Lreg)
    return loss, Ldist, Lreg, ce, alpha



# define function for EDL loss
def evidential_loss_from_batch(evidence, labels):
    """
    Negative Log-Likelihood (NLL) style EDL loss:
      sum_{k=1 to K}[ y_k * ( log( sum_j alpha_j ) - log(alpha_k) ) ]
    Returns:
      total_loss
    """
    alpha = evidence + 1.0  # alpha_k = e_k + 1
    alpha0 = alpha.sum(dim=1, keepdim=True)  # sum of all alpha_k for each sample

    K = alpha.shape[1]
    # Convert integer labels -> one-hot
    y_onehot = F.one_hot(labels, num_classes=K).float()

    # EDL NLL: \sum_k [ y_k * ( log(\sum_j alpha_j) - log(alpha_k) ) ]
    log_sum_alpha = torch.log(alpha0)
    log_alpha = torch.log(alpha)

    per_sample_loss = torch.sum(y_onehot * (log_sum_alpha - log_alpha), dim=1)
    loss = per_sample_loss.mean()
    return loss

# for evidential uncertainty from Dirichlet
def compute_evidential_unknown_score(evidence):
    """
    Optional helper if you want an 'unknown score' akin to 'compute_unknown_score' 
    in the variance-based approach. 
    Score = predictive uncertainty = K / sum(alpha_k).
    """
    alpha = evidence + 1.0
    alpha0 = alpha.sum(dim=1)   # shape (B,)
    K = alpha.shape[1]
    # The higher this is, the more uncertain => more likely unknown
    unknown_score = K / alpha0  # shape (B,)
    return unknown_score
