#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import copy
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

# 引入自定义 loss
from loss import variance_aware_loss_from_batch, scheduled_variance_aware_loss

class RAVDESSMultimodalDataset(Dataset):
    def __init__(self, samples, media_dir, num_frames, label_map, video_transform=None):
        self.media_dir = media_dir
        self.num_frames = num_frames
        self.label_map = label_map
        self.samples = [(a, v, label_map[lbl]) for (a, v, lbl) in samples]
        if video_transform is None:
            self.video_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.video_transform = video_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_fn, video_fn, label = self.samples[idx]
        wav_path = os.path.join(self.media_dir, audio_fn)
        mp4_path = os.path.join(self.media_dir, video_fn)
        waveform = load_audio_file(wav_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        frames = load_video_frames(
            mp4_path,
            num_frames=self.num_frames,
            transform=self.video_transform
        )
        return waveform, frames, label


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


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    optimizer,
    scheduler,
    device: torch.device,
    accelerator: Accelerator,
    loss_type: str,
    num_epochs: int = 10,
    lambda_reg: float = 0.02,
    T_acc: float = 0.98,
    P_acc: int = 2,
    eps_reg: float = 0.01,
    P_reg: int = 3,
    video_comb: int = 1,
    audio_comb: int = 1,
    weights_dir_visual: str = "weights/backbones/visual",
    weights_dir_audio: str = "weights/backbones/audio"
):
    # 加载 backbones
    v_path = os.path.join(weights_dir_visual, f"openset_split_combination_{video_comb}_timesformer_backbone.pth")
    a_path = os.path.join(weights_dir_audio, f"openset_split_combination_{audio_comb}_wav2vec_backbone.pth")
    backbone_v = load_timesformer_backbone(v_path, device)
    backbone_a = load_audio_backbone(a_path, device)

    compute_R = loss_type in ("variance", "scheduled")
    acc_counter = 0
    acc_saturated = False
    reg_counter = 0
    best_R = float('inf')
    best_R_wts = None

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    regs, delta_Rs = [], []

    for epoch in range(1, num_epochs+1):
        # 训练
        model.train()
        total_loss = total_correct = total_samples = 0
        for wavs, vids, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            wavs, vids, labels = wavs.to(device), vids.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feat_v = extract_frame_features_from_backbone(vids, backbone_v)
                feat_a = extract_audio_features_from_backbone(wavs, backbone_a)
            logits, z = model([feat_a, feat_v])
            items = loss_fn(logits, z, labels, epoch)
            loss = items[0]
            if compute_R and len(items) >= 3:
                Ldist = items[1].item()
                Lreg  = items[2].item()
                R = Ldist + lambda_reg * Lreg
            else:
                R = None
            accelerator.backward(loss)
            optimizer.step()
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        train_losses.append(total_loss/total_samples)
        train_accs.append(total_correct/total_samples)

        # 验证
        model.eval()
        v_loss = v_correct = v_samples = 0
        with torch.no_grad():
            for wavs, vids, labels in val_loader:
                wavs, vids, labels = wavs.to(device), vids.to(device), labels.to(device)
                feat_v = extract_frame_features_from_backbone(vids, backbone_v)
                feat_a = extract_audio_features_from_backbone(wavs, backbone_a)
                logits, z = model([feat_a, feat_v])
                items = loss_fn(logits, z, labels, epoch)
                loss = items[0]
                preds = logits.argmax(dim=1)
                v_loss += loss.item() * labels.size(0)
                v_correct += (preds == labels).sum().item()
                v_samples += labels.size(0)
        val_loss = v_loss/v_samples
        val_acc  = v_correct/v_samples
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 记录 R 和 ΔR
        if compute_R:
            regs.append(R)
            delta_Rs.append(regs[-2] - regs[-1] if len(regs)>1 else None)
        else:
            regs.append(None)
            delta_Rs.append(None)

        # 日志输出
        if accelerator.is_main_process:
            dR_str = f"{delta_Rs[-1]:.4f}" if delta_Rs[-1] is not None else "N/A"
            print(f"Epoch {epoch}/{num_epochs}: Train {train_losses[-1]:.4f}, Acc {train_accs[-1]:.4f} | "
                  f"Val {val_loss:.4f}, Acc {val_acc:.4f} | ΔR {dR_str}")

        # Open-Set Early Stop
        if not acc_saturated:
            if val_acc >= T_acc:
                acc_counter += 1
                if acc_counter >= P_acc:
                    acc_saturated = True
        else:
            if compute_R and len(delta_Rs) >= P_reg:
                if all(d is not None and d < eps_reg for d in delta_Rs[-P_reg:]):
                    if accelerator.is_main_process:
                        print(f"Open-Set early stopping at epoch {epoch}.")
                    break

        scheduler.step()

    # 加载最佳 R 权重
    if compute_R and best_R_wts is not None:
        model.load_state_dict(best_R_wts)

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "delta_Rs": delta_Rs
    }


def main():
    parser = argparse.ArgumentParser(description="Train multimodal fusion on RAVDESS")
    parser.add_argument("--csv_file", type=str, default="/media/data1/ningtong/wzh/datasets/RAVDESS/csv/multimodel/multimodal-combination-1.csv")
    parser.add_argument("--media_dir", type=str, default="/media/data1/ningtong/wzh/datasets/RAVDESS/data")
    parser.add_argument("--output_dir", type=str, default="/media/data1/ningtong/wzh/projects/Face-VII/weights")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_end", type=float, default=1e-6)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--video_comb", type=int, default=1)
    parser.add_argument("--audio_comb", type=int, default=1)
    parser.add_argument("--loss_type", choices=["ce","variance","scheduled"], default="scheduled")
    parser.add_argument("--lambda_reg", type=float, default=0.02)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--T_acc", type=float, default=0.98)
    parser.add_argument("--P_acc", type=int, default=2)
    parser.add_argument("--eps_reg", type=float, default=0.01)
    parser.add_argument("--P_reg", type=int, default=3)
    args = parser.parse_args()

    accelerator = Accelerator()
    raw_df = pd.read_csv(args.csv_file)
    label_map = {v: i for i,v in enumerate(sorted(raw_df['emo_label'].unique()))}
    train_samples = list(zip(*(raw_df[raw_df['category']=='train'][c] for c in ['audio_filename','video_filename','emo_label'])))
    val_df = raw_df[(raw_df['category']=='test')&(raw_df['emo_label']!=8)]
    val_samples = list(zip(*(val_df[c] for c in ['audio_filename','video_filename','emo_label'])))

    video_transform = T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_loader = DataLoader(
        RAVDESSMultimodalDataset(train_samples,args.media_dir,args.num_frames,label_map,video_transform),
        batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn_modality
    )
    val_loader = DataLoader(
        RAVDESSMultimodalDataset(val_samples,args.media_dir,args.num_frames,label_map,video_transform),
        batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn_modality
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalTransformer(modality_num=2,num_classes=len(label_map),num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr_end)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model,optimizer,train_loader,val_loader)

    if args.loss_type=="ce":
        loss_fn=lambda logits,z,labels,epoch:(nn.CrossEntropyLoss()(logits,labels),)
    elif args.loss_type=="variance":
        loss_fn=lambda logits,z,labels,epoch: variance_aware_loss_from_batch(z,logits,labels,lambda_reg=args.lambda_reg,lambda_cls=args.lambda_cls)
    else:
        loss_fn=lambda logits,z,labels,epoch: scheduled_variance_aware_loss(z,logits,labels,current_epoch=epoch,total_epochs=args.epochs,lambda_reg=args.lambda_reg,lambda_cls=args.lambda_cls)

    trained_model, metrics = train_and_evaluate(
        model,train_loader,val_loader,loss_fn,optimizer,scheduler,device,accelerator,
        args.loss_type,args.epochs,args.lambda_reg,
        args.T_acc,args.P_acc,args.eps_reg,args.P_reg,
        args.video_comb,args.audio_comb,
        os.path.join(args.output_dir,"backbones","visual"),
        os.path.join(args.output_dir,"backbones","audio")
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir,exist_ok=True)
        prefix = os.path.splitext(os.path.basename(args.csv_file))[0]
        filename = f"{prefix}_{args.loss_type}"
        save_path = os.path.join(args.output_dir,f"{filename}.pth")
        accelerator.save(trained_model.state_dict(),save_path)
        print(f"Saved model to {save_path}")

        epochs_ran = len(metrics['train_losses'])
        plt.figure()
        plt.plot(range(1,epochs_ran+1),metrics['train_losses'],label='Train Loss')
        plt.plot(range(1,epochs_ran+1),metrics['val_losses'],  label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.savefig(os.path.join(args.output_dir,f"{filename}_loss.png"))

        plt.figure()
        plt.plot(range(1,epochs_ran+1),metrics['train_accs'],label='Train Acc')
        plt.plot(range(1,epochs_ran+1),metrics['val_accs'],  label='Val Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        plt.savefig(os.path.join(args.output_dir,f"{filename}_acc.png"))

        plt.figure()
        delta_Rs = metrics['delta_Rs']
        if any(d is not None for d in delta_Rs):
            xs = [i+1 for i,d in enumerate(delta_Rs) if d is not None]
            ys = [d for d in delta_Rs if d is not None]
            plt.plot(xs, ys, label='ΔR')
            plt.xlabel('Epoch'); plt.ylabel('Delta R'); plt.legend()
        else:
            plt.text(0.5, 0.5, 'ΔR = N/A', ha='center', va='center')
            plt.axis('off')
        plt.savefig(os.path.join(args.output_dir,f"{filename}_deltaR.png"))

if __name__ == '__main__':
    main()