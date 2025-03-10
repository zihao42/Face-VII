from datetime import datetime
import os
import torch
import torch.nn as nn
from transformers import SwinForImageClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from loss import variance_aware_loss_from_batch, evidential_loss_from_batch, scheduled_variance_aware_loss
from enn_head import EvidentialClassificationHead

##############################
# 标签映射相关函数
##############################
def generate_label_map(uk):
    """
    根据传入的要舍弃的标签（chosen_label），生成映射字典。
    数据集原始标签范围为 0～6（共7个类别），
    舍弃 chosen_label 后，其余 6 个标签映射为连续索引 0～5。
    
    参数：
      chosen_label：要舍弃的标签（字符串或整数），例如 "3" 或 3
    返回：
      label_map：字典，例如若 chosen_label 为 3，则返回 {0:0, 1:1, 2:2, 4:3, 5:4, 6:5}
    """
    if uk == "sur":
        chosen_label = 0
    elif uk == "fea":
        chosen_label = 1
    elif uk == "dis":
        chosen_label = 2
    elif uk == "hap":
        chosen_label = 3
    elif uk == "sad":
        chosen_label = 4
    elif uk == "ang":
        chosen_label = 5
    elif uk == "neu":
        chosen_label = 6
    else:
        chosen_label = None
    
    valid_labels = [i for i in range(7) if i != chosen_label]
    label_map = {orig: new for new, orig in enumerate(sorted(valid_labels))}
    return label_map

def map_labels(labels, mapping):
    """
    根据 mapping 字典，将标签 tensor 中的每个值转换为连续的标签（类型为 torch.long）。
    """
    mapped = torch.tensor([mapping[int(x)] for x in labels.cpu().tolist()],
                            dtype=torch.long, device=labels.device)
    return mapped

##############################
# 辅助函数：计算均值与方差、未知评分
##############################
def compute_mu_var(z, labels, eps=1e-6):
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
    mu, var = compute_mu_var(z, labels, eps)
    B, M = z.shape
    z_expanded = z.unsqueeze(1)      # Shape: (B, 1, M)
    mu_expanded = mu.unsqueeze(0)      # Shape: (1, K', M)
    diff = z_expanded - mu_expanded
    normalized_squared = diff ** 2 / (var.unsqueeze(0) + eps)
    d = normalized_squared.sum(dim=2)  # 每个样本与各类别的 Mahalanobis 距离
    S, _ = d.min(dim=1)
    return S

##############################
# 训练函数
##############################
def train(num_epochs,
          eval_gap_epoch,
          num_labels,
          uk,
          dataloader_train,
          dataloader_eval,
          save_weights_gap_epoch,
          save_weight_dir,
          use_variance=False,
          use_schedule=False,
          evi=False,
          use_bn=False):
    """
    训练模型。
    
    参数：
      num_epochs：训练总轮数
      eval_gap_epoch：每隔多少个 epoch 进行一次评估
      num_labels：映射后类别数量，应为 6
      chosen_label：要舍弃的标签（原始数据中 0～6 中的一个），例如 "3" 或 3
      dataloader_train：训练数据加载器，返回的标签为原始标签（例如 0,1,2,4,5,6）
      dataloader_eval：验证数据加载器，同上
      save_weights_gap_epoch：每隔多少个 epoch 保存一次模型权重
      save_weight_dir：保存权重的目录
      use_variance, use_schedule, evi：不同模式的训练开关
      
    说明：
      数据加载器返回的原始标签保持不变（例如 0,1,2,4,5,6），
      本函数内部会生成一个映射字典，将这些标签转换为连续的 0～5，
      以满足 nn.CrossEntropyLoss 的要求，并确保模型输出与标签匹配。
    """
    # 生成标签映射字典，例如若 chosen_label 为 3，则映射为 {0:0, 1:1, 2:2, 4:3, 5:4, 6:5}
    label_map = generate_label_map(uk)
    print("生成的标签映射：", label_map)

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

    # 注册 hook，用于捕获进入分类器前的特征
    def hook_fn(module, input, output):
        pooled_representation["features"] = output.pooler_output

    model_name = "microsoft/swin-tiny-patch4-window7-224"
    # 这里使用 num_labels（映射后数量，应该为6）
    model = SwinForImageClassification.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
        num_labels=num_labels
    )
    model.swin.register_forward_hook(hook_fn)

    # define optimizer
    evi_head = None
    # define EDL head in evi mode
    if evi:
        in_features = model.config.hidden_size
        evi_head = EvidentialClassificationHead(in_features, num_labels, use_bn)

    if evi:
        # have to take account of the parameters of the evidential head
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(evi_head.parameters()), lr=5e-5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)
    model.to(device)
    # to device only in evi mode
    if evi:
        evi_head.to(device)

    # Tensorboard writer
    writer = SummaryWriter()

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # Training loop
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
            # 将原始标签转换为连续标签（例如原始标签 [0,1,2,4,5,6] 映射为 [0,1,2,3,4,5]）
            mapped_labels = map_labels(labels, label_map)
            # 断言映射后的标签在合法范围内
            assert torch.all((mapped_labels >= 0) & (mapped_labels < num_labels)), f"发现超出范围的标签：{mapped_labels}"
            
            optimizer.zero_grad()
            outputs = model(images).logits
            features = pooled_representation["features"]

            if use_variance:
                if use_schedule:
                    loss, ldist, lreg, ce_loss, _ = scheduled_variance_aware_loss(
                        features, outputs, mapped_labels, current_epoch=epoch, total_epochs=num_epochs,
                        lambda_reg=0.05, lambda_cls=1.0
                    )
                else:
                    loss, ldist, lreg, ce_loss = variance_aware_loss_from_batch(
                        features, outputs, mapped_labels, lambda_reg=0.05, lambda_cls=1.0
                    )
                total_ldist += ldist.item()
                total_lreg += lreg.item()
                total_ce_loss += ce_loss.item()
            elif evi:
                # for evidential
                evidence = evi_head(features)
                loss = evidential_loss_from_batch(evidence, mapped_labels)
            else:
                loss = nn.CrossEntropyLoss()(outputs, mapped_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if evi:
                evidence = evi_head(features)
                alpha = evidence + 1
                probs = alpha / alpha.sum(dim=1, keepdim=True)
                predicted = probs.argmax(dim=1)
            else:
                _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == mapped_labels).sum().item()
            total += mapped_labels.size(0)
        epoch_train_loss = total_loss / len(dataloader_train)
        epoch_train_acc = 100 * correct / total
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]: Train Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.2f}%")
        writer.add_scalar("Loss/total", epoch_train_loss, epoch)

        if use_variance:
            print(f"Loss_ldist: {total_ldist / len(dataloader_train):.4f}, "
                  f"Loss_lreg: {total_lreg / len(dataloader_train):.4f}, "
                  f"Loss_ce_loss: {total_ce_loss / len(dataloader_train):.4f}")
            writer.add_scalar("Loss/ldist", total_ldist / len(dataloader_train), epoch)
            writer.add_scalar("Loss/lreg", total_lreg / len(dataloader_train), epoch)
            writer.add_scalar("Loss/ce_loss", total_ce_loss / len(dataloader_train), epoch)

        # evaluation
        if (epoch + 1) % eval_gap_epoch == 0:
            model.eval()

            # for edl head if in evi mode
            if evi_head:
                evi_head.eval()
            
            test_correct = 0
            test_loss = 0
            test_total = 0
            test_ldist = 0
            test_lreg = 0
            test_ce_loss = 0
            with torch.no_grad():
                for images, labels in dataloader_eval:
                    images, labels = images.to(device), labels.to(device)
                    mapped_labels = map_labels(labels, label_map)
                    assert torch.all((mapped_labels >= 0) & (mapped_labels < num_labels)), f"Eval中发现超出范围的标签：{mapped_labels}"
                    
                    outputs = model(images).logits
                    features = pooled_representation["features"]
                    if use_variance:
                        if use_schedule:
                            losses = scheduled_variance_aware_loss(
                                features, outputs, mapped_labels, current_epoch=epoch, total_epochs=num_epochs,
                                lambda_reg=0.05, lambda_cls=1.0
                            )
                            test_loss += losses[0].item()
                            test_ldist += losses[1].item()
                            test_lreg += losses[2].item()
                            test_ce_loss += losses[3].item()
                        else:
                            losses = variance_aware_loss_from_batch(
                                features, outputs, mapped_labels, lambda_reg=0.05, lambda_cls=1.0
                            )
                            test_loss += losses[0].item()
                            test_ldist += losses[1].item()
                            test_lreg += losses[2].item()
                            test_ce_loss += losses[3].item()
                    elif evi:
                        evidence = evi_head(features)
                        loss = evidential_loss_from_batch(evidence, mapped_labels)
                        test_loss += loss.item()
                    else:
                        test_loss += nn.CrossEntropyLoss()(outputs, mapped_labels).item()

                    # for evi use edl head instead of Swin built-in logits
                    if evi:
                        evidence = evi_head(features)
                        alpha = evidence + 1
                        probs = alpha / alpha.sum(dim=1, keepdim=True)
                        predicted = probs.argmax(dim=1)
                    else:
                        _, predicted = torch.max(outputs, 1)

                    test_correct += (predicted == mapped_labels).sum().item()
                    test_total += mapped_labels.size(0)

            epoch_val_loss = test_loss / len(dataloader_eval)
            epoch_val_acc = 100 * test_correct / test_total
            val_loss_list.append(epoch_val_loss)
            val_acc_list.append(epoch_val_acc)
            print(f"\nEvaluation after Epoch {epoch + 1}: Eval Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_acc:.2f}%")
            writer.add_scalar("Loss/total_eval", epoch_val_loss, epoch)
            if use_variance:
                print(f"Loss_ldist: {test_ldist / len(dataloader_eval):.4f}, "
                      f"Loss_lreg: {test_lreg / len(dataloader_eval):.4f}, "
                      f"Loss_ce_loss: {test_ce_loss / len(dataloader_eval):.4f}\n")
                writer.add_scalar("Loss/ldist_eval", test_ldist / len(dataloader_eval), epoch)
                writer.add_scalar("Loss/lreg_eval", test_lreg / len(dataloader_eval), epoch)
                writer.add_scalar("Loss/ce_loss_eval", test_ce_loss / len(dataloader_eval), epoch)

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
    
    # plot_dir = "/media/data1/ningtong/wzh/projects/Face-VII/plot"
    # again, to run on colab
    plot_dir = "./plot"

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
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
