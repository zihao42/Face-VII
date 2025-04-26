# evaluation.py
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import AUROC
from torchmetrics.functional import auc as tm_auc
from sklearn.metrics import roc_curve
from data import get_openset_dataloaders
from audio_feature_extract import load_audio_backbone
from visual_feature_extract import load_timesformer_backbone
from feature_fusion import MultimodalTransformer
from predict import predict_batch, generate_label_map, inverse_label_map
from tqdm import tqdm

BATCH_SIZE = 32

def evaluate_combination(comb, data_dir,
                         video_dir, audio_dir,
                         clf_dir, csv_dir,
                         output_dir):
    # 找到对应的权重文件
    vid_w = glob.glob(os.path.join(video_dir, f"*combination_{comb}*.pth"))[0]
    aud_w = glob.glob(os.path.join(audio_dir, f"*combination_{comb}*.pth"))[0]
    clf_w = glob.glob(os.path.join(clf_dir, f"*combination-{comb}_*.pth"))[0]

    # 根据 classifier 权重文件名生成结果文件路径
    base = os.path.basename(clf_w)
    name = os.path.splitext(base)[0]
    results_txt = os.path.join(output_dir, f"evaluation-{name}.txt")

    print(f"\n=== Evaluating Combination {comb} ===")
    print(f"[INFO] Video weights: {os.path.basename(vid_w)}")
    print(f"[INFO] Audio weights: {os.path.basename(aud_w)}")
    print(f"[INFO] Fusion weights: {os.path.basename(clf_w)}")
    csv_path = os.path.join(csv_dir, f"multimodal-combination-{comb}.csv")
    print(f"[INFO] Using CSV file: {os.path.basename(csv_path)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_bb = load_timesformer_backbone(vid_w, device)
    audio_bb = load_audio_backbone(aud_w, device)
    label_map = generate_label_map(comb)
    inv_map = inverse_label_map(label_map)
    fusion_model = MultimodalTransformer(
        modality_num=2,
        num_classes=len(label_map),
        input_dim=video_bb.config.hidden_size,
        num_layers=2,
        feature_only=False
    )
    fusion_model.load_state_dict(torch.load(clf_w, map_location=device))
    fusion_model.to(device).eval()

    _, _, test_loader = get_openset_dataloaders(
        data_dir, csv_dir, comb,
        modality='both', batch_size=BATCH_SIZE
    )

    # AUROC: collect scores & labels
    auroc_metric = AUROC(pos_label=1)
    scores, labels = [], []
    for batch, gts in tqdm(test_loader, desc=f"Comb {comb} AUROC"):
        _, unk_scores, _ = predict_batch(
            batch['video'], batch['audio'],
            video_bb, audio_bb, fusion_model,
            label_map, inv_map,
            threshold=0.0
        )
        scores.extend(unk_scores)
        labels.extend([1 if gt == 8 else 0 for gt in gts.tolist()])
    scores_t = torch.tensor(scores)
    labels_t = torch.tensor(labels, dtype=torch.int)
    auroc = auroc_metric(scores_t, labels_t)
    print(f"AUROC: {auroc:.4f}")

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels_t.numpy(), scores_t.numpy())

    # OSCR: compute false positive rates & accuracies over thresholds
    frates, accs = [], []
    thr_range = torch.linspace(0.0, 1.0, steps=21)
    for t in tqdm(thr_range.tolist(), desc=f"Comb {comb} OSCR"):
        fp, kn_corr, kn_tot = 0, 0, 0
        for batch, gts in test_loader:
            preds, _, maxp = predict_batch(
                batch['video'], batch['audio'],
                video_bb, audio_bb, fusion_model,
                label_map, inv_map,
                threshold=t
            )
            for mp, pr, gt in zip(maxp.tolist(), preds, gts.tolist()):
                if gt != 8:
                    kn_tot += 1
                    if mp < t:
                        fp += 1
                    elif pr == gt:
                        kn_corr += 1
        frates.append(fp / kn_tot if kn_tot else 0.0)
        accs.append(kn_corr / kn_tot if kn_tot else 0.0)
    fr_t = torch.tensor(frates)
    ac_t = torch.tensor(accs)
    oscr_auc = tm_auc(fr_t, ac_t)
    print(f"OSCR AUC: {oscr_auc:.4f}")

    # 写入 evaluation 结果文件
    if not os.path.exists(results_txt):
        with open(results_txt, "w") as f:
            f.write("combination,AUROC,OSCR_AUC\n")
    with open(results_txt, "a") as f:
        f.write(f"{comb},{auroc:.4f},{oscr_auc:.4f}\n")

    # 绘制当前组合的 OSCR 曲线
    plt.plot(frates, accs, label=f"Comb {comb}")

    return fpr, tpr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/media/data1/ningtong/wzh/datasets/RAVDESS/data")
    parser.add_argument("--video_weights_dir", default="/media/data1/ningtong/wzh/projects/Face-VII/weights/backbones/visual")
    parser.add_argument("--audio_weights_dir", default="/media/data1/ningtong/wzh/projects/Face-VII/weights/backbones/audio")
    parser.add_argument("--classifier_weights_dir", required=True)
    parser.add_argument("--csv_dir", default="/media/data1/ningtong/wzh/datasets/RAVDESS/csv/multimodel-reduced")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save plots and result logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 收集所有组合的 ROC 曲线数据
    roc_curves = {}

    # 绘制并保存 OSCR 曲线
    plt.figure()
    for comb in range(1, 11):
        fpr, tpr = evaluate_combination(
            comb,
            args.data_dir,
            args.video_weights_dir,
            args.audio_weights_dir,
            args.classifier_weights_dir,
            args.csv_dir,
            args.output_dir
        )
        roc_curves[comb] = (fpr, tpr)

    plt.xlabel("False Positive Rate (known→unknown)")
    plt.ylabel("Known class accuracy")
    plt.title("OSCR Curves for all combinations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    oscr_fig = os.path.join(args.output_dir, "oscr_curves.png")
    plt.savefig(oscr_fig)
    print(f"[INFO] OSCR curves saved to {oscr_fig}")
    plt.close()

    # 绘制并保存 ROC 曲线汇总图
    plt.figure()
    for comb, (fpr, tpr) in roc_curves.items():
        plt.plot(fpr, tpr, alpha=0.3, label=f"Comb {comb}")
    # 平均 ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs_interp = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curves.values()]
    mean_tpr = np.mean(tprs_interp, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='red', linewidth=2, label="Mean ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for all combinations")
    plt.legend()
    plt.grid(True)
    roc_fig = os.path.join(args.output_dir, "roc_curves.png")
    plt.savefig(roc_fig)
    print(f"[INFO] ROC curves saved to {roc_fig}")
    plt.close()

    # 保存 ROC 数据到 txt
    roc_data_txt = os.path.join(args.output_dir, "roc_data.txt")
    with open(roc_data_txt, "w") as f:
        for comb, (fpr, tpr) in roc_curves.items():
            f.write(f"Combination {comb}\n")
            f.write("FPR: " + ",".join(map(str, fpr.tolist())) + "\n")
            f.write("TPR: " + ",".join(map(str, tpr.tolist())) + "\n\n")
        f.write("Mean ROC\n")
        f.write("Mean FPR: " + ",".join(map(str, mean_fpr.tolist())) + "\n")
        f.write("Mean TPR: " + ",".join(map(str, mean_tpr.tolist())) + "\n")
    print(f"[INFO] ROC data saved to {roc_data_txt}")

if __name__ == "__main__":
    main()
