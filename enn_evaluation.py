# evaluation.py

import os, glob
import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import AUROC
from torchmetrics.functional import auc as tm_auc
from data import get_openset_dataloaders
from predict import load_models  # reuse loading & ENN logic

BATCH_SIZE = 4

def evaluate_comb(comb, data_dir, v_dir, a_dir, c_dir, e_dir, csv_dir):
    print(f"\n=== Split {comb} ===")
    # locate weight files
    v_w = glob.glob(os.path.join(v_dir, f"*{comb}*.pth"))[0]
    a_w = glob.glob(os.path.join(a_dir, f"*{comb}*.pth"))[0]
    c_w = glob.glob(os.path.join(c_dir, f"*{comb}*.pth"))[0]
    e_w = glob.glob(os.path.join(e_dir, f"*{comb}*.pth"))[0]
    # load models
    v_bb, a_bb, clf, enn, label_map, inv_map, device = load_models(
        comb, v_w, a_w, c_w, e_w
    )
    # data
    _, _, test_loader = get_openset_dataloaders(
        data_dir, csv_dir, comb,
        modality='both', batch_size=BATCH_SIZE
    )
    # 1) AUROC on vacuity scores
    auroc = AUROC(pos_label=1)
    scores, labels = [], []
    for batch, gts in test_loader:
        vids, auds = batch['video'].to(device), batch['audio'].to(device)
        with torch.no_grad():
            vf = extract_video_features(vids, v_bb).mean(dim=1)
            af = extract_audio_features(auds, a_bb, target_frames=vf.shape[1]).mean(dim=1)
            fused = torch.cat([vf, af], dim=1)
            evidence = enn(fused)
            alpha = evidence + 1.0
            S = alpha.sum(dim=1)
            vac = (alpha.shape[1] / S).cpu().tolist()
        scores += vac
        labels += [1 if gt==8 else 0 for gt in gts.tolist()]
    auroc_val = auroc(torch.tensor(scores), torch.tensor(labels))
    print(f"AUROC (vacuity): {auroc_val:.4f}")

    # 2) OSCR sweep vacuity threshold
    fprs, accs = [], []
    for t in torch.linspace(0,1,51).tolist():
        fp=0; kc=0; kt=0
        for batch, gts in test_loader:
            vids, auds = batch['video'].to(device), batch['audio'].to(device)
            with torch.no_grad():
                vf = extract_video_features(vids, v_bb).mean(dim=1)
                af = extract_audio_features(auds, a_bb, target_frames=vf.shape[1]).mean(dim=1)
                fused = torch.cat([vf, af], dim=1)
                logits = clf(fused)
                evidence = enn(fused)
                alpha = evidence + 1
                S = alpha.sum(dim=1)
                vac = (alpha.shape[1]/S).cpu().tolist()
                probs = alpha / alpha.sum(dim=1,keepdim=True)
                _, preds = probs.max(dim=1)
                origs = [inv_map[p.item()] for p in preds]
            for v_score, pr, gt in zip(vac, origs, gts.tolist()):
                if gt!=8:
                    kt+=1
                    if v_score>t:
                        fp+=1
                    elif pr==gt:
                        kc+=1
        fprs.append(fp/kt if kt else 0.0)
        accs.append(kc/kt if kt else 0.0)
    oscr = tm_auc(torch.tensor(fprs), torch.tensor(accs))
    print(f"OSCR AUC: {oscr:.4f}")
    plt.plot(fprs, accs, label=f"Split {comb}")

def main():
    import argparse
    p = argparse.ArgumentParser("Evaluate all splits")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--video_weights_dir", required=True)
    p.add_argument("--audio_weights_dir", required=True)
    p.add_argument("--classifier_weights_dir", required=True)
    p.add_argument("--enn_weights_dir", required=True)
    p.add_argument("--csv_dir", required=True)
    args = p.parse_args()

    plt.figure()
    for c in range(1,11):
        evaluate_comb(
            c,
            args.data_dir,
            args.video_weights_dir,
            args.audio_weights_dir,
            args.classifier_weights_dir,
            args.enn_weights_dir,
            args.csv_dir
        )
    plt.xlabel("FPR (known→unknown)")
    plt.ylabel("Known Accuracy")
    plt.title("OSCR Curves (splits 1–10)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
