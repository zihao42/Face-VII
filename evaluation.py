# evaluation.py
import os
import glob
import torch
import matplotlib.pyplot as plt
from torchmetrics import AUROC
from torchmetrics.functional import auc as tm_auc
from data import get_openset_dataloaders, COMBINATION_SPLITS
from predict import load_models, predict_batch

BATCH_SIZE = 4


def evaluate_combination(comb, data_dir,
                         video_dir, audio_dir,
                         clf_dir, csv_dir):
    # find weight files by combination
    vid_w = glob.glob(os.path.join(video_dir, f"*{comb}*.pth"))[0]
    aud_w = glob.glob(os.path.join(audio_dir, f"*{comb}*.pth"))[0]
    clf_w = glob.glob(os.path.join(clf_dir, f"*{comb}*.pth"))[0]
    print(f"\n=== Combination {comb} ===")
    # load models
    video_bb, audio_bb, classifier, label_map, inv_map, device = load_models(
        comb, vid_w, aud_w, clf_w
    )
    # get test loader
    _, _, test_loader = get_openset_dataloaders(
        data_dir, csv_dir, comb,
        modality='both', batch_size=BATCH_SIZE
    )
    # collect scores for AUROC
    auroc_metric = AUROC(pos_label=1)
    scores, labels = [], []
    for batch, gts in test_loader:
        _, unk_scores, _ = predict_batch(
            batch['video'], batch['audio'],
            video_bb, audio_bb, classifier,
            label_map, inv_map,
            threshold=0.0  # ignore threshold for score collection
        )
        scores.extend(unk_scores)
        labels.extend([1 if gt==8 else 0 for gt in gts.tolist()])
    # compute AUROC
    scores_t = torch.tensor(scores)
    labels_t = torch.tensor(labels, dtype=torch.int)
    auroc = auroc_metric(scores_t, labels_t)
    print(f"AUROC: {auroc:.4f}")
    # compute OSCR curve
    frates, accs = [], []
    thr_range = torch.linspace(0.0, 1.0, steps=51)
    for t in thr_range.tolist():
        fp, kn_corr, kn_tot = 0, 0, 0
        for batch, gts in test_loader:
            preds, _, maxp = predict_batch(
                batch['video'], batch['audio'],
                video_bb, audio_bb, classifier,
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
        frates.append(fp/kn_tot if kn_tot else 0.0)
        accs.append(kn_corr/kn_tot if kn_tot else 0.0)
    fr_t = torch.tensor(frates)
    ac_t = torch.tensor(accs)
    oscr_auc = tm_auc(fr_t, ac_t)
    print(f"OSCR AUC: {oscr_auc:.4f}")
    # plot
    plt.plot(frates, accs, label=f"Comb {comb}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Root folder of multimodal data (mp4, wav)")
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
