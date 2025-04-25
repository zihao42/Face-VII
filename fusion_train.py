#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import re
import copy
import traceback
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm import tqdm
from accelerate import Accelerator

from data import load_video_frames, load_audio_file
from feature_fusion import MultimodalTransformer
from audio_feature_extract import load_audio_backbone, extract_audio_features_from_backbone
from visual_feature_extract import load_timesformer_backbone, extract_frame_features_from_backbone
from enn_head import EvidentialClassificationHead
# 引入自定义 loss
from loss import variance_aware_loss_from_batch, scheduled_variance_aware_loss, evidential_loss_from_batch


def collate_fn_modality(batch):
    waveforms, videos, labels = zip(*batch)
    max_len = max(wf.shape[1] for wf in waveforms)
    padded = []
    for wf in waveforms:
        pad_len = max_len - wf.shape[1]
        wf_p = F.pad(wf, (0, pad_len), mode='constant', value=0)
        if wf_p.dim() == 2:
            wf_p = wf_p.squeeze(0)
        padded.append(wf_p)
    audio_batch = torch.stack(padded, dim=0)
    video_batch = torch.stack(videos, dim=0)
    label_batch = torch.tensor(labels, dtype=torch.long)
    return audio_batch, video_batch, label_batch


class RAVDESSMultimodalDataset(Dataset):
    def __init__(self, samples, media_dir, num_frames, label_map, video_transform=None):
        self.samples = [(a, v, label_map[lbl]) for (a, v, lbl) in samples]
        self.media_dir = media_dir
        self.num_frames = num_frames
        self.video_transform = video_transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a_fn, v_fn, lbl = self.samples[idx]
        wav = load_audio_file(os.path.join(self.media_dir, a_fn))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        frames = load_video_frames(
            os.path.join(self.media_dir, v_fn),
            num_frames=self.num_frames,
            transform=self.video_transform
        )
        return wav, frames, lbl


def train_and_evaluate(
    model, train_loader, val_loader, loss_fn,
    optimizer, scheduler, device, accelerator,
    loss_type, num_epochs,
    lambda_reg, lambda_cls,
    P_acc, P_reg,
    video_comb, audio_comb,
    weights_dir_visual, weights_dir_audio,
    lr_end,         # 新增：最小学习率
    val_threshold,  # 新增：精度触发阈值
    enn_head = None
):
    # 最早开始监控的 epoch
    MIN_MONITOR_EPOCH = 3

    # R 模式下的忽略区间：R 在此区间时，不更新权重、不计数器、不早停
    IGNORE_R_LOW  = 1.2
    IGNORE_R_HIGH = 1.8

    # 载入预训练 backbone
    backbone_v = load_timesformer_backbone(
        os.path.join(weights_dir_visual,
                     f"openset_split_combination_{video_comb}_timesformer_backbone.pth"),
        device
    )
    backbone_a = load_audio_backbone(
        os.path.join(weights_dir_audio,
                     f"openset_split_combination_{audio_comb}_wav2vec_backbone.pth"),
        device
    )

    compute_R = loss_type in ("variance", "scheduled")

    # 最优 checkpoint & epoch
    best_ckpt_wts    = None
    best_epoch       = None
    best_R           = float('inf')
    best_val_acc_ce  = 0.0
    best_val_loss_ce = float('inf')
    # 新增：用于 R 模式下的验证准确率监控
    best_val_acc_R   = 0.0

    # Early-Stop 计数
    val_loss_counter = 0
    reg_counter      = 0
    ce_acc_no_imp    = 0
    ce_acc_req       = 0.03
    # 新增：R 模式下的 val_acc 计数器
    val_acc_counter  = 0

    aborted = False
    train_losses, val_losses = [], []
    train_accs, val_accs     = [], []
    Rs                       = []

    # 控制何时直接使用 lr_end
    use_min_lr = False

    try:
        for epoch in range(1, num_epochs + 1):
            # 1. 计算 alpha（仅 scheduled）
            alpha = None
            if loss_type == "scheduled":
                alpha = 0.0 if epoch == 1 else ((train_accs[-1] + val_accs[-1]) / 2) ** 2

            # 2. 训练阶段
            model.train()
            tot_loss = tot_corr = tot_samps = 0
            for wavs, vids, labels in tqdm(
                    train_loader,
                    desc=f"Train Epoch {epoch}/{num_epochs}",
                    disable=not accelerator.is_main_process,
                    leave=False):
                wavs, vids, labels = wavs.to(device), vids.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    feat_v = extract_frame_features_from_backbone(vids, backbone_v)
                    feat_a = extract_audio_features_from_backbone(wavs, backbone_a)
                if enn_head is not None:
                    fused_feat = model([feat_a, feat_v])
                    evidence = enn_head(fused_feat)
                    loss = evidential_loss_from_batch(evidence, labels)
                    alpha = evidence + 1.0
                    probs = alpha / alpha.sum(dim=1, keepdim=True)
                    preds = torch.argmax(probs, dim=1)
                    # as enn loss only return 1 item, won't use R mode
                    items = loss_fn(evidence, labels)
                else:
                    logits, z = model([feat_a, feat_v])
                    items = (
                        loss_fn(logits, z, labels, alpha)
                        if loss_type == "scheduled"
                        else loss_fn(logits, z, labels, epoch)
                    )
                    loss = items[0]
                    preds = logits.argmax(dim=1)

                if compute_R and len(items) >= 3:
                    R = items[1].item() + lambda_reg * items[2].item()
                else:
                    R = None
            
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                tot_loss  += loss.item() * labels.size(0)
                tot_corr  += (preds == labels).sum().item()
                tot_samps += labels.size(0)

            train_losses.append(tot_loss / tot_samps)
            train_accs.append(tot_corr / tot_samps)

            # 3. 验证阶段
            model.eval()
            v_loss = v_corr = v_samps = 0
            for wavs, vids, labels in tqdm(
                    val_loader,
                    desc=f"Val   Epoch {epoch}/{num_epochs}",
                    disable=not accelerator.is_main_process,
                    leave=False):
                wavs, vids, labels = wavs.to(device), vids.to(device), labels.to(device)
                with torch.no_grad():
                    feat_v = extract_frame_features_from_backbone(vids, backbone_v)
                    feat_a = extract_audio_features_from_backbone(wavs, backbone_a)
    
                if enn_head is not None:
                    fused_feat = model([feat_a, feat_v])
                    evidence = enn_head(fused_feat)
                    loss = evidential_loss_from_batch(evidence, labels)
                    alpha = evidence + 1.0
                    probs = alpha / alpha.sum(dim=1, keepdim=True)
                    preds = torch.argmax(probs, dim=1)
                else:
                    logits, z = model([feat_a, feat_v])
                    items = (
                            loss_fn(logits, z, labels, alpha)
                        if loss_type == "scheduled"
                        else loss_fn(logits, z, labels, epoch)
                    )
                    loss = items[0]
                    preds = logits.argmax(dim=1)

                v_loss  += loss.item() * labels.size(0)
                v_corr  += (preds == labels).sum().item()
                v_samps += labels.size(0)

            val_loss = v_loss / v_samps
            val_acc  = v_corr / v_samps
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            Rs.append(R)

            # 4. 日志
            if accelerator.is_main_process:
                r_str = f"{R:.4f}" if R is not None else "N/A"
                print(f"Epoch {epoch}/{num_epochs}: "
                      f"Train Loss {train_losses[-1]:.4f}, Acc {train_accs[-1]:.4f} | "
                      f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f} | R {r_str}")

            # --- 5/6. 从 MIN_MONITOR_EPOCH 轮才开始更新最优 & Early-Stop ---
            if epoch >= MIN_MONITOR_EPOCH:
                # Pure-CE 模式
                if R is None:
                    if val_loss < best_val_loss_ce:
                        best_val_loss_ce = val_loss
                        best_ckpt_wts    = copy.deepcopy(model.state_dict())
                        best_epoch       = epoch
                        val_loss_counter = 0
                    else:
                        val_loss_counter += 1

                    if val_acc >= best_val_acc_ce + ce_acc_req:
                        best_val_acc_ce = val_acc
                        best_ckpt_wts   = copy.deepcopy(model.state_dict())
                        best_epoch      = epoch
                        ce_acc_no_imp   = 0
                    else:
                        ce_acc_no_imp += 1

                    if ce_acc_no_imp >= P_acc:
                        if accelerator.is_main_process:
                            print(f"Early stopping (CE style) at epoch {epoch}.")
                        break
                else:
                    # 如果 R 在忽略区间内，则跳过本轮监控
                    if IGNORE_R_LOW < R < IGNORE_R_HIGH:
                        if accelerator.is_main_process:
                            print(f"Epoch {epoch}: R={R:.4f} is abnormal，epoch ignored.")
                    else:
                        # 正式的 R 模式更新 & 计数
                        if R < best_R:
                            best_R        = R
                            best_ckpt_wts = copy.deepcopy(model.state_dict())
                            best_epoch    = epoch
                            reg_counter   = 0
                        else:
                            reg_counter += 1

                        if val_acc >= best_val_acc_R:
                            best_val_acc_R = val_acc
                            val_acc_counter = 0
                        elif val_acc >= best_val_acc_R - ce_acc_req:
                            val_acc_counter = 0
                        else:
                            val_acc_counter += 1

                        # Early-Stop 判定
                        if val_acc_counter >= P_acc or reg_counter >= P_reg:
                            if accelerator.is_main_process:
                                print(f"Early stopping (R style) at epoch {epoch}.")
                            break

            # 在验证后，根据阈值决定学习率
            if not use_min_lr and val_acc >= val_threshold:
                use_min_lr = True
                if accelerator.is_main_process:
                    print(f"Val Acc reached {val_acc:.4f} ≥ {val_threshold:.2f}, switch to lr_end={lr_end}")

            # 7. 更新 lr
            if use_min_lr:
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_end
            else:
                scheduler.step()

    except Exception:
        aborted = True
        if accelerator.is_main_process:
            traceback.print_exc()
            print("Training aborted due to exception.")

    # 恢复最优
    if best_ckpt_wts is not None:
        model.load_state_dict(best_ckpt_wts)

    metrics = {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "train_accs":   train_accs,
        "val_accs":     val_accs,
        "Rs":           Rs,
        "best_epoch":   best_epoch,
        "aborted":      aborted
    }
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train multimodal fusion on RAVDESS")
    parser.add_argument("--csv_file",      type=str,
                        default="/media/data1/ningtong/wzh/datasets/RAVDESS/csv/multimodel/multimodal-combination-1.csv")
    parser.add_argument("--media_dir",     type=str,
                        default="/media/data1/ningtong/wzh/datasets/RAVDESS/data")
    parser.add_argument("--output_dir",    type=str,
                        default="/media/data1/ningtong/wzh/projects/Face-VII/weights")
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--epochs",        type=int,   default=25)
    parser.add_argument("--lr",            type=float, default=5e-6)
    parser.add_argument("--lr_end",        type=float, default=2e-6)
    parser.add_argument("--val_threshold", type=float, default=0.8,
                        help="当 val_acc ≥ 阈值时直接切换到 lr_end")
    parser.add_argument("--num_frames",    type=int,   default=32)
    parser.add_argument("--video_comb",    type=int,   default=None)
    parser.add_argument("--audio_comb",    type=int,   default=None)
    parser.add_argument("--loss_type",     choices=["ce","variance","scheduled", "evi"],
                        default="scheduled")
    parser.add_argument("--lambda_reg",    type=float, default=0.7)
    parser.add_argument("--lambda_cls",    type=float, default=0.6)
    parser.add_argument("--P_acc",         type=int,   default=5)
    parser.add_argument("--P_reg",         type=int,   default=4)
    args = parser.parse_args()

    # 自动从 csv_file 名称中提取 combination 编号
    basename = os.path.basename(args.csv_file)
    m = re.search(r'combination[-_]?(\d+)\.csv$', basename)
    if m:
        comb = int(m.group(1))
        args.video_comb = comb if args.video_comb is None else args.video_comb
        args.audio_comb = comb if args.audio_comb is None else args.audio_comb
    else:
        raise ValueError(f"无法从 csv 文件名 ‘{basename}’ 中识别 combination 编号")

    # 打印要加载的文件路径
    visual_wpath = os.path.join(
        args.output_dir, "backbones", "visual",
        f"openset_split_combination_{args.video_comb}_timesformer_backbone.pth"
    )
    audio_wpath = os.path.join(
        args.output_dir, "backbones", "audio",
        f"openset_split_combination_{args.audio_comb}_wav2vec_backbone.pth"
    )
    print(f"----------------------------------------------------------------------------")
    print(f"CSV file:        {args.csv_file}")
    print(f"Visual backbone: {visual_wpath}")
    print(f"Audio backbone:  {audio_wpath}")
    print(f"----------------------------------------------------------------------------")

    accelerator = Accelerator()

    raw_df = pd.read_csv(args.csv_file)
    unique = sorted(raw_df['emo_label'].unique())
    label_map = {lbl: i for i, lbl in enumerate(unique)}

    train_samples = list(zip(
        raw_df[raw_df.category=="train"].audio_filename,
        raw_df[raw_df.category=="train"].video_filename,
        raw_df[raw_df.category=="train"].emo_label
    ))
    val_df = raw_df[(raw_df.category=="test") & (raw_df.emo_label!=8)]
    val_samples = list(zip(
        val_df.audio_filename, val_df.video_filename, val_df.emo_label
    ))

    video_transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_loader = DataLoader(
        RAVDESSMultimodalDataset(train_samples, args.media_dir,
                                 args.num_frames, label_map, video_transform),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_modality
    )
    val_loader = DataLoader(
        RAVDESSMultimodalDataset(val_samples, args.media_dir,
                                 args.num_frames, label_map, video_transform),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_modality
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalTransformer(
        modality_num=2, num_classes=len(label_map), num_layers=2, feature_only=(args.loss_type == "evi")
    ).to(device)

    # enn head
    if args.loss_type == "evi":
        enn_head = EvidentialClassificationHead(model.embed_dim * model.n_modality, len(label_map), use_bn=True).to(device)
    else:  
        enn_head = None

    # optimizer
    if args.loss_type == "evi":
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(enn_head.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_end
    )

    model, train_loader, val_loader, optimizer = accelerator.prepare(
        model, train_loader, val_loader, optimizer
    )

    if args.loss_type == "ce":
        loss_fn = lambda logits, z, labels, *_: (
            nn.CrossEntropyLoss()(logits, labels),)
    elif args.loss_type == "variance":
        loss_fn = lambda logits, z, labels, *_: variance_aware_loss_from_batch(
            z, logits, labels,
            lambda_reg=args.lambda_reg,
            lambda_cls=args.lambda_cls
        )
    elif args.loss_type == "evi":
        loss_fn = lambda evidence, labels: evidential_loss_from_batch(evidence, labels)
    else:
        enn_head = None
        loss_fn = lambda logits, z, labels, epoch: scheduled_variance_aware_loss(z, logits, labels, current_epoch=epoch, total_epochs=args.epochs, lambda_reg=args.lambda_reg, lambda_cls=args.lambda_cls)

    # 训练
    trained_model, metrics = train_and_evaluate(
        model, train_loader, val_loader, loss_fn,
        optimizer, scheduler, device, accelerator,
        args.loss_type, args.epochs,
        args.lambda_reg, args.lambda_cls,
        args.P_acc, args.P_reg,
        args.video_comb, args.audio_comb,
        os.path.join(args.output_dir, "backbones", "visual"),
        os.path.join(args.output_dir, "backbones", "audio"),
        args.lr_end,
        args.val_threshold,
        enn_head=enn_head
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        prefix   = os.path.splitext(os.path.basename(args.csv_file))[0]
        filename = f"{prefix}_{args.loss_type}"
        if args.loss_type == "evi":
            accelerator.save({
                    "model": trained_model.state_dict(),
                    "enn_head": enn_head.state_dict()
                }, os.path.join(args.output_dir, f"{filename}.pth"))
        else:
            accelerator.save(trained_model.state_dict(),
                             os.path.join(args.output_dir, f"{filename}.pth"))
        print(f"Saved model to {filename}.pth")

        best_epoch  = metrics["best_epoch"]
        aborted     = metrics["aborted"]
        epochs_ran  = len(metrics["train_losses"])

        # Loss 曲线
        plt.figure()
        plt.plot(range(1, epochs_ran+1), metrics["train_losses"], label="Train Loss")
        plt.plot(range(1, epochs_ran+1), metrics["val_losses"],   label="Val Loss")
        plt.axvline(best_epoch, color='red', linestyle='--', label='Best Epoch')
        if aborted: plt.title("Training aborted due to exception")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.savefig(os.path.join(args.output_dir, f"{filename}_loss.png"))

        # Acc 曲线
        plt.figure()
        plt.plot(range(1, epochs_ran+1), metrics["train_accs"], label="Train Acc")
        plt.plot(range(1, epochs_ran+1), metrics["val_accs"],   label="Val Acc")
        plt.axvline(best_epoch, color='red', linestyle='--', label='Best Epoch')
        if aborted: plt.title("Training aborted due to exception")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
        plt.savefig(os.path.join(args.output_dir, f"{filename}_acc.png"))

        # R 曲线
        plt.figure()
        Rs = metrics["Rs"]
        xs = [i+1 for i, r in enumerate(Rs) if r is not None]
        ys = [r     for r in Rs if r is not None]
        if xs:
            plt.plot(xs, ys, label="R")
            plt.axvline(best_epoch, color='red', linestyle='--', label='Best Epoch')
            if aborted: plt.title("Training aborted due to exception")
            plt.xlabel("Epoch"); plt.ylabel("R"); plt.legend()
        else:
            plt.text(0.5,0.5,"R = N/A", ha="center", va="center")
            if aborted: plt.title("Training aborted due to exception")
            plt.axis("off")
        plt.savefig(os.path.join(args.output_dir, f"{filename}_R.png"))

if __name__ == "__main__":
    main()
