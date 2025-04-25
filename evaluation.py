# evaluation.py
import os
import glob
import torch
import matplotlib.pyplot as plt
from torchmetrics import AUROC
from torchmetrics.functional import auc as tm_auc
from data import get_openset_dataloaders, COMBINATION_SPLITS, generate_label_map, inverse_label_map
from audio_feature_extract import load_audio_backbone
from visual_feature_extract import load_timesformer_backbone
from feature_fusion import MultimodalTransformer
from predict import predict_batch  # ← Import batch predictor

BATCH_SIZE = 4

def evaluate_combination(comb, data_dir,
                         video_dir, audio_dir,
                         clf_dir, csv_dir):
    vid_w = glob.glob(os.path.join(video_dir, f"*combination_{comb}*.pth"))[0]
    aud_w = glob.glob(os.path.join(audio_dir, f"*combination_{comb}*.pth"))[0]
    clf_w = glob.glob(os.path.join(clf_dir, f"*combination-{comb}_scheduled.pth"))[0]

    print(f"\n=== Evaluating Combination {comb} ===")
    print(f"[INFO] Video weights: {os.path.basename(vid_w)}")
    print(f"[INFO] Audio weights: {os.path.basename(aud_w)}")
    print(f"[INFO] Fusion weights: {os.path.basename(clf_w)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_bb = load_timesformer_backbone(vid_w, device)
    audio_bb = load_audio_backbone(aud_w, device)
    label_map = generate_label_map(comb)
    inv_map = inverse_label_map(label_map)
    fusion_model = MultimodalTransformer(
        modality_num=2,
        num_classes=len(label_map),
        input_dim=video_bb.config.hidden_size,
        feature_only=False
    )
    fusion_model.load_state_dict(torch.load(clf_w, map_location=device))
    fusion_model.to(device).eval()

    _, _, test_loader = get_openset_dataloaders(
        data_dir, csv_dir, comb,
        modality='both', batch_size=BATCH_SIZE
    )

    # AUROC
    auroc_metric = AUROC(pos_label=1)
    scores, labels = [], []
    for batch, gts in test_loader:
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

    # OSCR
    frates, accs = [], []
    thr_range = torch.linspace(0.0, 1.0, steps=51)
    for t in thr_range.tolist():
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

    plt.plot(frates, accs, label=f"Comb {comb}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--video_weights_dir", required=True)
    parser.add_argument("--audio_weights_dir", required=True)
    parser.add_argument("--classifier_weights_dir", required=True)
    parser.add_argument("--csv_dir", required=True)
    args = parser.parse_args()

    plt.figure()
    for comb in range(1, 11):
        evaluate_combination(
            comb,
            args.data_dir,
            args.video_weights_dir,
            args.audio_weights_dir,
            args.classifier_weights_dir,
            args.csv_dir
        )

    plt.xlabel("False Positive Rate (known→unknown)")
    plt.ylabel("Known class accuracy")
    plt.title("OSCR Curves for all combinations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
