#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation_noevi_closeset.py: Close-set evaluation for CE multimodal models on RAVDESS.

输出：
  - 每个 combination 的混淆矩阵 CSV 文件： combination-<id>_confusion.csv
  - 平均混淆矩阵 CSV 文件： average_confusion.csv
  - 平均混淆矩阵热力图： average_confusion.png
  - 每个 combination 的 accuracy、precision、recall 及其平均汇总： metrics_summary.csv

用法：
  python evaluation_noevi_closeset.py \
    --weights_dir /path/to/model-ce \
    --audio_backbone_dir /path/to/backbones/audio \
    --visual_backbone_dir /path/to/backbones/visual \
    --csv_dir /path/to/csvs/multimodel-reduced \
    --media_dir /path/to/RAVDESS/data \
    --batch_size 32 \
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from fusion_train import RAVDESSMultimodalDataset, collate_fn_modality  # 数据集与 collate_fn 复用训练脚本
from feature_fusion import MultimodalTransformer
from audio_feature_extract import load_audio_backbone, extract_audio_features_from_backbone
from visual_feature_extract import load_timesformer_backbone, extract_frame_features_from_backbone


def parse_combination_number(filename):
    m = re.search(r'combination[-_]?(\d+)', filename)
    if not m:
        raise ValueError(f"Cannot parse combination number from {filename}")
    return int(m.group(1))


def evaluate_single_combination_closeset(
    comb_id, clf_path,
    audio_bb_path, visual_bb_path,
    csv_path, args
):
    print(f"\n--- Combination {comb_id} (close-set) ---")
    print(f"  Classifier:      {clf_path}")
    print(f"  Audio backbone:  {audio_bb_path}")
    print(f"  Visual backbone: {visual_bb_path}")
    print(f"  CSV file:        {csv_path}")

    # 1) 构建标签映射：train ∪ test_known
    raw_df   = pd.read_csv(csv_path)
    train_df = raw_df[raw_df.category == "train"]
    val_df   = raw_df[(raw_df.category == "test") & (raw_df.emo_label != 8)]
    labels_train  = set(train_df['emo_label'])
    labels_val    = set(val_df['emo_label'])
    unique_lbls   = sorted(labels_train.union(labels_val))
    label_map     = {lbl: idx for idx, lbl in enumerate(unique_lbls)}
    num_classes   = len(label_map)
    
    print("Label Map:", label_map)
    print("Unique Labels:", unique_lbls)
    
    # 2) 构造测试样本，只取 test & emo_label != 8
    test_df = val_df
    samples = list(zip(
        test_df.audio_filename.values,
        test_df.video_filename.values,
        test_df.emo_label.values
    ))

    # 3) 加载数据集
    dataset = RAVDESSMultimodalDataset(
        samples,
        args.media_dir,
        args.num_frames,
        label_map
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_modality,
        num_workers=4,
        pin_memory=(args.device.startswith("cuda"))
    )

    # 4) 加载 backbones + 分类器
    device = torch.device(args.device)
    backbone_v = load_timesformer_backbone(visual_bb_path, device)
    backbone_a = load_audio_backbone(audio_bb_path, device)

    model = MultimodalTransformer(
        modality_num=2,
        num_classes=num_classes,
        num_layers=2,
        feature_only=False
    ).to(device)
    state = torch.load(clf_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 5) 推理并收集标签索引
    y_true_idx, y_pred_idx = [], []
    with torch.no_grad():
        for wavs, vids, lbls in tqdm(loader, desc=f"Comb {comb_id}", leave=False):
            wavs = wavs.to(device)
            vids = vids.to(device)
            feat_v = extract_frame_features_from_backbone(vids, backbone_v)
            feat_a = extract_audio_features_from_backbone(wavs, backbone_a)
            logits, _ = model([feat_a, feat_v])
            preds = logits.argmax(dim=1).cpu().numpy()

            for raw_lbl, p in zip(lbls, preds):
                raw_int = int(raw_lbl.item()) if hasattr(raw_lbl, "item") else int(raw_lbl.item())
                y_true_idx.append(raw_int)
                y_pred_idx.append(int(p))

    y_true_idx = np.array(y_true_idx, dtype=int)
    y_pred_idx = np.array(y_pred_idx, dtype=int)

    # 6) 计算混淆矩阵（5×5）
    cm = confusion_matrix(
        y_true_idx,
        y_pred_idx,
        labels=list(range(num_classes))
    )
    return cm, unique_lbls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir',        type=str, required=True,
                        help='CE 模型权重目录')
    parser.add_argument('--audio_backbone_dir',  default="/media/data1/ningtong/wzh/projects/Face-VII/weights/backbones/audio")
    parser.add_argument('--visual_backbone_dir', default="/media/data1/ningtong/wzh/projects/Face-VII/weights/backbones/visual")
    parser.add_argument('--csv_dir',             default="/media/data1/ningtong/wzh/datasets/RAVDESS/csv/multimodel-reduced")
    parser.add_argument('--media_dir',           default="/media/data1/ningtong/wzh/datasets/RAVDESS/data")
    parser.add_argument('--batch_size',          type=int, default=32)
    parser.add_argument('--num_frames',          type=int, default=32)
    parser.add_argument('--device',              type=str, default='cuda')
    parser.add_argument('--output_dir',          type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA 不可用，切换到 CPU")
        args.device = 'cpu'

    # 遍历 CE 权重
    all_clfs = sorted(f for f in os.listdir(args.weights_dir) if f.endswith('.pth'))
    # 准备全 8 类标签列表
    full_labels = list(range(8))
    cms = []
    metrics_list = []

    for clf in all_clfs:
        try:
            comb_id = parse_combination_number(clf)
        except ValueError:
            continue

        clf_path = os.path.join(args.weights_dir, clf)
        audio_bb_path = os.path.join(
            args.audio_backbone_dir,
            f"openset_split_combination_{comb_id}_wav2vec_backbone.pth"
        )
        visual_bb_path = os.path.join(
            args.visual_backbone_dir,
            f"openset_split_combination_{comb_id}_timesformer_backbone.pth"
        )
        csv_path = os.path.join(
            args.csv_dir,
            f"multimodal-combination-{comb_id}.csv"
        )

        # 计算单个 combination 的 5×5 混淆矩阵
        cm, labels = evaluate_single_combination_closeset(
            comb_id, clf_path,
            audio_bb_path, visual_bb_path,
            csv_path, args
        )

        # 将 5×5 矩阵 zero‐pad 到 8×8
        cm8 = np.zeros((8, 8), dtype=cm.dtype)
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                cm8[li, lj] = cm[i, j]
        cms.append(cm8)

        # 计算宏平均准确率、精确率、召回率
        total_correct = cm.trace()
        total = cm.sum()
        accuracy = total_correct / total if total > 0 else 0.0
        precisions = []
        recalls = []
        for k in range(len(labels)):
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        metrics_list.append({
            'combination': comb_id,
            'accuracy': accuracy,
            'precision': np.mean(precisions),
            'recall': np.mean(recalls)
        })

    # 计算平均混淆矩阵并保存
    avg_cm8 = np.mean(cms, axis=0)
    df_avg8 = pd.DataFrame(avg_cm8, index=full_labels, columns=full_labels)
    df_avg8.to_csv(os.path.join(args.output_dir, "average_confusion.csv"))

    # 绘制并保存热力图
    plt.figure()
    cm_vals = df_avg8.values
    im = plt.imshow(
        cm_vals,
        cmap='Blues',
        vmin=0,
        vmax=cm_vals.max(),
        interpolation='nearest'
    )
    plt.title("Average Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    # 在每个格子中添加数值注释，高对比度显示
    thresh = cm_vals.max() / 2.0
    for i in range(cm_vals.shape[0]):
        for j in range(cm_vals.shape[1]):
            plt.text(j, i, f"{cm_vals[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if cm_vals[i, j] > thresh else "black")
    plt.xticks(np.arange(len(full_labels)), full_labels, rotation=45)
    plt.yticks(np.arange(len(full_labels)), full_labels)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "average_confusion.png"))
    plt.close()

    # 保存 metrics 汇总 CSV
    df_metrics = pd.DataFrame(metrics_list).set_index('combination')
    df_metrics.loc['average'] = df_metrics.mean()
    df_metrics.to_csv(
        os.path.join(args.output_dir, "metrics_summary.csv"),
        index_label='combination'
    )

    print(f"\nAll results saved in {args.output_dir}")




if __name__ == "__main__":
    main()
