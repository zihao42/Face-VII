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

def scheduled_variance_aware_loss(z, logits, labels, current_epoch, total_epochs, lambda_reg=0.02, lambda_cls=1.0):
    eps = 1e-6
    min_var = 1e-1
    unique_labels = labels.unique()
    
    class_stats = {}
    for cl in unique_labels:
        cl_mask = (labels == cl)
        z_cl = z[cl_mask]
        mu_cl = z_cl.mean(dim=0)
        var_cl = torch.clamp(z_cl.var(dim=0, unbiased=False) + eps, min=min_var)
        class_stats[int(cl.item())] = (mu_cl, var_cl)
    
    mu_batch = torch.stack([class_stats[int(lbl.item())][0] for lbl in labels], dim=0)
    var_batch = torch.stack([class_stats[int(lbl.item())][1] for lbl in labels], dim=0)
    
    diff = z - mu_batch
    Ldist = 0.5 * torch.sum(diff * diff / var_batch, dim=1).mean()
    
    # 使用 hinge 方式的正则化：只惩罚 var < 1
    Lreg = torch.mean(torch.relu(-torch.log(var_batch)))
    
    ce_loss = nn.CrossEntropyLoss()(logits, labels)
    
    # 调度因子：前半程仅用分类损失，后半程采用二次 ramp-up 平滑引入 Ldist 和 Lreg
    half_epoch = total_epochs / 2.0
    if current_epoch < half_epoch:
        alpha = 0.0
    else:
        linear_alpha = (current_epoch - half_epoch) / half_epoch
        alpha = min(linear_alpha**2, 1.0)
    
    loss = lambda_cls * ce_loss + alpha * (Ldist + lambda_reg * Lreg)
    return loss, Ldist, Lreg, ce_loss, alpha


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
