#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation_evi.py: Evaluate evidential multimodal models on RAVDESS open-set test.

Usage:
    python evaluation_evi.py \
        --weights_dir /path/to/evi_checkpoints \
        --audio_backbone_dir /path/to/backbones/audio \
        --visual_backbone_dir /path/to/backbones/visual \
        --csv_dir /path/to/csvs/multimodel-reduced \
        --media_dir /path/to/media \
        --batch_size 32 \
        --threshold 0.5 \
        --num_frames 32 \
        --device cuda \
        --output_dir /path/to/results
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score


from data import load_audio_file, load_video_frames
from fusion_train import collate_fn_modality
from enn_predict import load_models, predict_batch

class EvaluationDataset(Dataset):
    """Dataset for evaluation: returns raw labels (0-8)."""
    def __init__(self, samples, media_dir, num_frames, video_transform=None):
        self.samples = samples
        self.media_dir = media_dir
        self.num_frames = num_frames
        from torchvision import transforms as T
        self.video_transform = video_transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a_fn, v_fn, raw_lbl = self.samples[idx]
        wav = load_audio_file(os.path.join(self.media_dir, a_fn))
        if wav.dim() > 1:
            wav = wav.mean(dim=0, keepdim=True)
        frames = load_video_frames(
            os.path.join(self.media_dir, v_fn),
            num_frames=self.num_frames,
            transform=self.video_transform
        )
        return wav, frames, raw_lbl


def parse_combination_number(filename):
    m = re.search(r'combination[-_]?(\d+)', filename)
    if not m:
        raise ValueError(f"Cannot parse combination from {filename}")
    return int(m.group(1))


def evaluate_single_combination_evi(
    comb_id, checkpoint_path,
    audio_backbone_path, visual_backbone_path,
    csv_path, args, roc_list, oscr_list
):
    print(f"\n=== Evaluating Combination {comb_id} (EVI) ===")
    print(f" Audio backbone:  {audio_backbone_path}")
    print(f" Visual backbone: {visual_backbone_path}")
    print(f" Checkpoint:      {checkpoint_path}")
    print(f" CSV file:        {csv_path}")

    device = args.device

    # 1) Load pretrained backbones and evidential model
    video_bb, audio_bb, fusion_model, label_map, inv_map, device, enn_head = \
        load_models(
            comb_id,
            visual_backbone_path,
            audio_backbone_path,
            checkpoint_path
        )

    # 2) Prepare data & mappings
    df       = pd.read_csv(csv_path)
    train_df = df[df.category == 'train']
    test_df  = df[df.category == 'test']
    known_labels = sorted(set(train_df.emo_label))
    map_known    = {lbl: i for i, lbl in enumerate(known_labels)}
    unknown_id   = len(map_known)

    samples = list(zip(
        test_df.audio_filename.values,
        test_df.video_filename.values,
        test_df.emo_label.values
    ))
    dataset = EvaluationDataset(samples, args.media_dir, args.num_frames)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_modality,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    y_true        = []
    raw_preds_arg = []
    confidences   = []

    # 3) Collect raw argmax preds & confidences (1 - vacuity)
    with torch.no_grad():
        for wavs, vids, raw_lbls in tqdm(loader, desc=f"Comb {comb_id}"):
            preds, vacs, _ = predict_batch(
                vids, wavs,
                (video_bb, audio_bb, fusion_model,
                 label_map, inv_map, device, enn_head),
                threshold=args.threshold
            )
            confs = [1.0 - v for v in vacs]

            for i, raw in enumerate(raw_lbls):
                raw_int = int(raw) if isinstance(raw, (int, np.integer)) else int(raw.item())
                gt = map_known[raw_int] if raw_int != 8 else unknown_id
                y_true.append(gt)
                raw_preds_arg.append(preds[i])
                confidences.append(confs[i])

    y_true      = np.array(y_true, dtype=int)
    raw_preds   = np.array(raw_preds_arg, dtype=int)
    confidences = np.array(confidences, dtype=float)

    # 4) Multi-threshold binary metrics (known vs unknown)
    bin_thr_vals = np.arange(0.05, 1.0, 0.1)
    bin_metrics  = []
    y_true_bin   = (y_true == unknown_id).astype(int)
    for thr in bin_thr_vals:
        y_pred_thr = np.where(confidences < thr, unknown_id, raw_preds)
        y_pred_bin = (y_pred_thr == unknown_id).astype(int)
        acc  = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        bin_metrics.append((thr, acc, prec, rec))

    # 5) Open-set AUROC & ROC
    y_os  = (y_true != unknown_id).astype(int)
    auroc = roc_auc_score(y_os, confidences)
    fpr, tpr, _ = roc_curve(y_os, confidences)
    roc_list.append((comb_id, fpr, tpr))

    # 6) OSCR: dynamic threshold with raw_preds_arg
    thr_vals = np.linspace(0, 1, 101)
    oscr_fprs, oscr_ccrs = [], []
    n_known   = (y_true != unknown_id).sum()
    n_unknown = (y_true == unknown_id).sum()
    for thr in thr_vals:
        detect = confidences >= thr
        correct_known = ((y_true != unknown_id) &
                         detect &
                         (raw_preds == y_true)).sum()
        ccr = correct_known / n_known if n_known > 0 else 0.0

        false_alarm = ((y_true == unknown_id) &
                       detect).sum()
        fpr_u = false_alarm / n_unknown if n_unknown > 0 else 0.0

        oscr_ccrs.append(ccr)
        oscr_fprs.append(fpr_u)

    idx = np.argsort(oscr_fprs)
    fprs_sorted = np.array(oscr_fprs)[idx].tolist()
    ccrs_sorted = np.array(oscr_ccrs)[idx].tolist()
    oscr = np.trapz(ccrs_sorted, fprs_sorted)
    oscr_list.append((comb_id, fprs_sorted, ccrs_sorted))

    # 7) Save results
    base    = f"multimodal-combination-{comb_id}_evi"
    os.makedirs(args.output_dir, exist_ok=True)
    res_path = os.path.join(args.output_dir, base + "_results.txt")
    with open(res_path, 'w') as f:
        f.write(f"Combination {comb_id} (EVI) Binary known-vs-unknown @ thresholds:\n")
        f.write("thr\tAccuracy\tPrecision\tRecall\n")
        for thr, acc, prec, rec in bin_metrics:
            f.write(f"{thr:.2f}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\n")
        f.write("\n")
        f.write(f"Open-set AUROC: {auroc:.4f}\n")
        f.write(f"OSCR          : {oscr:.4f}\n")

    # Save ROC & OSCR data
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
        os.path.join(args.output_dir, base + '_roc.csv'), index=False
    )
    pd.DataFrame({'fpr': fprs_sorted, 'ccr': ccrs_sorted}).to_csv(
        os.path.join(args.output_dir, base + '_oscr.csv'), index=False
    )

    print(f"Saved: {res_path}")
    print(f"Saved: {base}_roc.csv")
    print(f"Saved: {base}_oscr.csv")

    return auroc, oscr

def plot_and_save_aggregate(roc_list, oscr_list, out_dir):
    # Aggregate ROC
    plt.figure()
    for comb, fpr, tpr in roc_list:
        plt.plot(fpr, tpr, alpha=0.3, label=f"Comb {comb}")
    grid = np.linspace(0, 1, 1000)
    interp_tprs = [np.interp(grid, fpr, tpr) for _, fpr, tpr in roc_list]
    mean_tpr = np.mean(interp_tprs, axis=0)
    plt.plot(grid, mean_tpr, color='red', linewidth=2, label='Mean ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Aggregate ROC (EVI)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'evi_aggregate_roc.png'))
    plt.close()

    # Aggregate OSCR
    plt.figure()
    for comb, fprs, ccrs in oscr_list:
        plt.plot(fprs, ccrs, alpha=0.3, label=f"Comb {comb}")
    interp_ccrs = [np.interp(grid, fprs, ccrs) for _, fprs, ccrs in oscr_list]
    mean_ccr = np.mean(interp_ccrs, axis=0)
    plt.plot(grid, mean_ccr, color='red', linewidth=2, label='Mean OSCR')
    plt.xlabel('False Positive Rate (Unknown)')
    plt.ylabel('Closed-set Classification Rate')
    plt.title('Aggregate OSCR (EVI)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'evi_aggregate_oscr.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', type=str, required=True,
                        help='Classifier weights directory')
    parser.add_argument('--audio_backbone_dir', type=str, default="/media/data1/ningtong/wzh/projects/Face-VII/weights/backbones/audio",
                        help='Audio backbone weights directory')
    parser.add_argument('--visual_backbone_dir', type=str, default="/media/data1/ningtong/wzh/projects/Face-VII/weights/backbones/visual",
                        help='Visual backbone weights directory')
    parser.add_argument('--csv_dir', type=str, default="/media/data1/ningtong/wzh/datasets/RAVDESS/csv/multimodel-reduced",
                        help='Directory of CSV files')
    parser.add_argument('--media_dir', type=str, default="/media/data1/ningtong/wzh/datasets/RAVDESS/data",
                        help='Directory of media data (audio/video)')
    parser.add_argument('--batch_size',        type=int, default=32)
    parser.add_argument('--threshold',         type=float, default=0.5)
    parser.add_argument('--num_frames',        type=int, default=32)
    parser.add_argument('--device',            default='cuda')
    parser.add_argument('--output_dir',        required=True)
    args = parser.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    clf_files = sorted(f for f in os.listdir(args.weights_dir) if f.endswith('.pth'))
    roc_list, oscr_list = [], []
    for clf in clf_files:
        try:
            comb_id = parse_combination_number(clf)
        except ValueError:
            continue
        chkpt = os.path.join(args.weights_dir, clf)
        aud_bb = os.path.join(args.audio_backbone_dir, f"openset_split_combination_{comb_id}_wav2vec_backbone.pth")
        vis_bb = os.path.join(args.visual_backbone_dir, f"openset_split_combination_{comb_id}_timesformer_backbone.pth")
        csv_path = os.path.join(args.csv_dir, f"multimodal-combination-{comb_id}.csv")
        evaluate_single_combination_evi(
            comb_id, chkpt, aud_bb, vis_bb, csv_path,
            args, roc_list, oscr_list
        )

    if roc_list and oscr_list:
        plot_and_save_aggregate(roc_list, oscr_list, args.output_dir)

if __name__ == '__main__':
    main()
