from datetime import datetime
import os
import torch
import torch.nn as nn
from transformers import SwinForImageClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from loss import variance_aware_loss_from_batch, scheduled_variance_aware_loss

def compute_mu_var(z, labels, eps=1e-6):
    """
    Compute the mean (mu) and diagonal variance (var) for each class based on the 
    current batch features and true labels.
    
    Args:
      z: Tensor of shape (B, M) representing the feature vectors of the current batch.
      labels: Tensor of shape (B,) representing the true labels.
      eps: A small constant for numerical stability.
      
    Returns:
      mu_tensor: Tensor of shape (K', M) with the mean vector for each class present in the batch.
      var_tensor: Tensor of shape (K', M) with the variance vector for each class.
    """
    unique_labels = torch.unique(labels)
    mu_list = []
    var_list = []
    for cl in sorted(unique_labels.tolist()):
        cl_mask = (labels == cl)
        z_cl = z[cl_mask]
        mu_cl = z_cl.mean(dim=0)
        var_cl = z_cl.var(dim=0, unbiased=False) + eps
        mu_list.append(mu_cl)
        var_list.append(var_cl)
    mu_tensor = torch.stack(mu_list, dim=0)
    var_tensor = torch.stack(var_list, dim=0)
    return mu_tensor, var_tensor

def compute_unknown_score(z, labels, eps=1e-6):
    """
    Compute the unknown score S for each sample based on the Mahalanobis distance.
    
    Args:
      z: Tensor of shape (B, M) representing the feature vectors.
      labels: Tensor of shape (B,) representing the true labels.
      eps: A small constant for numerical stability.
      
    Returns:
      S: Tensor of shape (B,) representing the unknown score for each sample.
    """
    mu, var = compute_mu_var(z, labels, eps)
    B, M = z.shape
    z_expanded = z.unsqueeze(1)  # Shape: (B, 1, M)
    mu_expanded = mu.unsqueeze(0)  # Shape: (1, K', M)
    diff = z_expanded - mu_expanded
    normalized_squared = diff ** 2 / (var.unsqueeze(0) + eps)
    d = normalized_squared.sum(dim=2)  # Mahalanobis distance for each sample with each class
    S, _ = d.min(dim=1)
    return S

def train(num_epochs,
          eval_gap_epoch,
          num_labels,
          dataloader_train,
          dataloader_eval,
          save_weights_gap_epoch,
          save_weight_dir,
          use_variance=False,
          use_schedule=False,
          evi=False):
    """
    Train the model with different loss functions based on the specified mode.
    
    Modes:
      - Baseline: use_variance=False, standard cross-entropy loss.
      - Only Variance: use_variance=True, use_schedule=False, use variance_aware_loss_from_batch.
      - Variance with Schedule: use_variance=True, use_schedule=True, use scheduled_variance_aware_loss.
      - Evidential: evi=True.
    """
    if use_variance:
        if use_schedule:
            print("Training with Variance and Schedule!")
        else:
            print("Training with Variance (without Schedule)!")
    elif evi:
        print("Evidential version training starts!")
    else:
        print("Baseline training starts (standard cross-entropy loss)!")

    date_time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    pooled_representation = {}

    # Hook function to capture the input to the classification layer.
    def hook_fn(module, input, output):
        pooled_representation["features"] = output.pooler_output  # Extract features before the FC layer

    # Load the pre-trained Swin-Tiny model.
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    model = SwinForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True,
                                                         num_labels=num_labels)
    model.swin.register_forward_hook(hook_fn)

    # Define the optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: " + str(device))
    model.to(device)
    writer = SummaryWriter()

    # Lists to record training and validation metrics.
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total = 0
        total_ldist = 0
        total_lreg = 0
        total_ce_loss = 0
        correct = 0

        for images, labels in tqdm(dataloader_train, desc="Processing"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            features = pooled_representation["features"]

            if use_variance:
                if use_schedule:
                    # 使用 scheduled_variance_aware_loss
                    loss, ldist, lreg, ce_loss, _ = scheduled_variance_aware_loss(
                        features, outputs, labels, current_epoch=epoch, total_epochs=num_epochs,
                        lambda_reg=0.05, lambda_cls=1.0
                    )
                else:
                    # 使用 variance_aware_loss_from_batch（无 schedule）
                    loss, ldist, lreg, ce_loss = variance_aware_loss_from_batch(
                        features, outputs, labels, lambda_reg=0.05, lambda_cls=1.0
                    )
                total_ldist += ldist.item()
                total_lreg += lreg.item()
                total_ce_loss += ce_loss.item()
            else:
                # Baseline：仅使用交叉熵损失
                loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        epoch_train_loss = total_loss / len(dataloader_train)
        epoch_train_acc = 100 * correct / total
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}]:")
        print(f"  Train Loss: {epoch_train_loss:.4f}")
        print(f"  Train Accuracy: {epoch_train_acc:.2f}%")
        writer.add_scalar("Loss/total", epoch_train_loss, epoch)

        if use_variance:
            print(
                f"  Loss_ldist: {total_ldist / len(dataloader_train):.4f}, "
                f"Loss_lreg: {total_lreg / len(dataloader_train):.4f}, "
                f"Loss_ce_loss: {total_ce_loss / len(dataloader_train):.4f}"
            )
            writer.add_scalar("Loss/ldist", total_ldist / len(dataloader_train), epoch)
            writer.add_scalar("Loss/lreg", total_lreg / len(dataloader_train), epoch)
            writer.add_scalar("Loss/ce_loss", total_ce_loss / len(dataloader_train), epoch)

        # Evaluation 每隔 eval_gap_epoch 个 epoch
        if (epoch + 1) % eval_gap_epoch == 0:
            model.eval()
            test_correct = 0
            test_loss = 0
            test_total = 0
            test_ldist = 0
            test_lreg = 0
            test_ce_loss = 0
            with torch.no_grad():
                for images, labels in dataloader_eval:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images).logits
                    features = pooled_representation["features"]
                    if use_variance:
                        if use_schedule:
                            losses = scheduled_variance_aware_loss(
                                features, outputs, labels, current_epoch=epoch, total_epochs=num_epochs,
                                lambda_reg=0.05, lambda_cls=1.0
                            )
                            test_loss += losses[0].item()
                            test_ldist += losses[1].item()
                            test_lreg += losses[2].item()
                            test_ce_loss += losses[3].item()
                        else:
                            losses = variance_aware_loss_from_batch(
                                features, outputs, labels, lambda_reg=0.05, lambda_cls=1.0
                            )
                            test_loss += losses[0].item()
                            test_ldist += losses[1].item()
                            test_lreg += losses[2].item()
                            test_ce_loss += losses[3].item()
                    else:
                        test_loss += nn.CrossEntropyLoss()(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    test_correct += (predicted == labels).sum().item()
                    test_total += labels.size(0)
            epoch_val_loss = test_loss / len(dataloader_eval)
            epoch_val_acc = 100 * test_correct / test_total
            val_loss_list.append(epoch_val_loss)
            val_acc_list.append(epoch_val_acc)
            print(f"Evaluation after Epoch {epoch + 1}:")
            print(f"  Eval Loss: {epoch_val_loss:.4f}")
            print(f"  Eval Accuracy: {epoch_val_acc:.2f}%")
            writer.add_scalar("Loss/total_eval", epoch_val_loss, epoch)
            if use_variance:
                print(
                    f"  Loss_ldist: {test_ldist / len(dataloader_eval):.4f}, "
                    f"Loss_lreg: {test_lreg / len(dataloader_eval):.4f}, "
                    f"Loss_ce_loss: {test_ce_loss / len(dataloader_eval):.4f}"
                )
                writer.add_scalar("Loss/ldist_eval", test_ldist / len(dataloader_eval), epoch)
                writer.add_scalar("Loss/lreg_eval", test_lreg / len(dataloader_eval), epoch)
                writer.add_scalar("Loss/ce_loss_eval", test_ce_loss / len(dataloader_eval), epoch)

        # 定期保存模型权重
        if (epoch + 1) % save_weights_gap_epoch == 0 and epoch + 1 < num_epochs:
            option = ""
            if use_variance:
                if use_schedule:
                    option = "_variance_schedule"
                else:
                    option = "_variance"
            elif evi:
                option = "_evidential"
            torch.save(model.state_dict(),
                       os.path.join(save_weight_dir,
                                    "swin_tiny_rafdb_" + date_time_str + "_epoch_" + str(epoch) + option + ".pth"))
    # 保存最终模型权重
    option = ""
    if use_variance:
        if use_schedule:
            option = "_variance_schedule"
        else:
            option = "_variance"
    elif evi:
        option = "_evidential"
    torch.save(model.state_dict(),
               os.path.join(save_weight_dir, "swin_tiny_rafdb_" + date_time_str + "_final" + option + ".pth"))
    
    # 生成并保存训练和验证指标图
    plot_dir = "/media/data1/ningtong/wzh/projects/Face-VII/plot"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # 绘制训练和验证 Loss 曲线
    plt.figure()
    epochs_range = range(1, num_epochs + 1)
    plt.plot(epochs_range, train_loss_list, label="Train Loss")
    plt.plot(epochs_range, val_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "loss_plot.png"))
    plt.close()
    
    # 绘制训练和验证 Accuracy 曲线
    plt.figure()
    plt.plot(epochs_range, train_acc_list, label="Train Accuracy")
    plt.plot(epochs_range, val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "accuracy_plot.png"))
    plt.close()
    
    writer.close()
