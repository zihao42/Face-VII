# evaluation.py

import os
import glob
import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import AUROC
from torchmetrics.functional import auc as tm_auc
from data import get_openset_dataloaders
from predict import load_models, predict_batch

BATCH_SIZE = 4


def evaluate_combination(comb, data_dir,
                         video_dir, audio_dir,
                         fusion_dir,
                         csv_dir):
    print(f"\n=== Combination {comb} ===")
    # patterns
    v_w = glob.glob(os.path.join(video_dir, f"*combination_{comb}*.pth"))[0]
    a_w = glob.glob(os.path.join(audio_dir, f"*combination_{comb}*.pth"))[0]
    f_w = glob.glob(os.path.join(fusion_dir, f"*combination-{comb}_scheduled.pth"))[0]
    print(f"Video weights : {v_w}")
    print(f"Audio weights : {a_w}")
    print(f"Fusion weights: {f_w}")

    # load models once
    models = load_models(comb, v_w, a_w, f_w)

    # dataloader
    _, _, test_loader = get_openset_dataloaders(
        data_dir, csv_dir, comb,
        modality='both', batch_size=BATCH_SIZE
    )

    # AUROC on vacuity
    print("Computing AUROC...")
    auroc = AUROC(pos_label=1)
    all_scores, all_labels = [], []
    for batch, gts in test_loader:
        vids, auds = batch['video'], batch['audio']
        _, vacs, _ = predict_batch(vids, auds, models, threshold=0.0)
        all_scores.extend(vacs)
        all_labels.extend([1 if gt==8 else 0 for gt in gts.tolist()])
    auroc_val = auroc(torch.tensor(all_scores), torch.tensor(all_labels))
    print(f"AUROC (vacuity): {auroc_val:.4f}")

    # OSCR curve
    print("Computing OSCR...")
    fprs, accs = [], []
    for t in torch.linspace(0,1,51).tolist():
        fp=kn_corr=kn_tot=0
        for batch, gts in test_loader:
            vids, auds = batch['video'], batch['audio']
            preds, vacs, _ = predict_batch(vids, auds, models, threshold=t)
            for v_score, pr, gt in zip(vacs, preds, gts.tolist()):
                if gt!=8:
                    kn_tot+=1
                    if v_score>t:
                        fp+=1
                    elif pr==gt:
                        kn_corr+=1
        fprs.append(fp/kn_tot if kn_tot else 0)
        accs.append(kn_corr/kn_tot if kn_tot else 0)
    oscr = tm_auc(torch.tensor(fprs), torch.tensor(accs))
    print(f"OSCR AUC: {oscr:.4f}")
    plt.plot(fprs, accs, label=f"Split {comb}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--video_weights_dir", required=True)
    p.add_argument("--audio_weights_dir", required=True)
    p.add_argument("--fusion_weights_dir", required=True)
    p.add_argument("--csv_dir", required=True)
    args = p.parse_args()

    plt.figure()
    for comb in range(1,11):
        evaluate_combination(
            comb,
            args.data_dir,
            args.video_weights_dir,
            args.audio_weights_dir,
            args.fusion_weights_dir,
            args.csv_dir
        )
    plt.xlabel("FPR (known→unknown)")
    plt.ylabel("Known Accuracy")
    plt.title("OSCR Curves (1–10)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
