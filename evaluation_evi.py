import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Data loading & model helpers
from data import load_audio_file, load_video_frames
from fusion_train import collate_fn_modality
from enn_predict import load_models, predict_batch

class EvaluationDataset(Dataset):
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

    video_bb, audio_bb, fusion_model, label_map, inv_map, device, enn_head = \
        load_models(
            comb_id,
            visual_backbone_path,
            audio_backbone_path,
            checkpoint_path
        )

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
        dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn_modality,
        num_workers=4, pin_memory=(device.type=='cuda')
    )

    y_true, raw_preds, confidences = [], [], []

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
                raw_int = int(raw) if isinstance(raw,(int,np.integer)) else int(raw.item())
                gt = map_known[raw_int] if raw_int != 8 else unknown_id
                y_true.append(gt)
                raw_preds.append(preds[i])
                confidences.append(confs[i])

    y_true      = np.array(y_true, dtype=int)
    raw_preds   = np.array(raw_preds, dtype=int)
    confidences = np.array(confidences, dtype=float)

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

    y_os        = (y_true != unknown_id).astype(int)
    auroc       = roc_auc_score(y_os, confidences)
    fpr, tpr, _ = roc_curve(y_os, confidences)
    roc_list.append((comb_id, fpr, tpr))

    thr_vals    = np.linspace(0, 1, 101)
    oscr_fprs   = []
    oscr_ccrs   = []
    n_known     = (y_true != unknown_id).sum()
    n_unknown   = (y_true == unknown_id).sum()
    for thr in thr_vals:
        detect       = confidences >= thr
        correct_kn   = ((y_true != unknown_id) & detect & (raw_preds == y_true)).sum()
        ccr          = correct_kn / n_known if n_known>0 else 0.0
        false_alarm  = ((y_true == unknown_id) & detect).sum()
        fpr_u        = false_alarm / n_unknown if n_unknown>0 else 0.0
        oscr_ccrs.append(ccr)
        oscr_fprs.append(fpr_u)
    idx            = np.argsort(oscr_fprs)
    fprs_sorted    = np.array(oscr_fprs)[idx].tolist()
    ccrs_sorted    = np.array(oscr_ccrs)[idx].tolist()
    oscr_value     = np.trapz(ccrs_sorted, fprs_sorted)
    oscr_list.append((comb_id, fprs_sorted, ccrs_sorted))

    comb_name = f"multimodal-combination-{comb_id}_evi"
    out_dir   = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # results.txt
    res_path = os.path.join(out_dir, f"{comb_name}_results.txt")
    with open(res_path, 'w') as f:
        f.write(f"Combination {comb_id} (EVI) Binary known-vs-unknown @ thresholds:\n")
        f.write("thr\tAccuracy\tPrecision\tRecall\n")
        for thr, acc, prec, rec in bin_metrics:
            f.write(f"{thr:.2f}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\n")
        f.write("\n")
        f.write(f"AUROC (open-set): {auroc:.4f}\n")
        f.write(f"OSCR            : {oscr_value:.4f}\n")

    # ROC & OSCR data TXT
    np.savetxt(
        os.path.join(out_dir, f"{comb_name}_roc_data.txt"),
        np.vstack([fpr, tpr]).T,
        header="fpr tpr", fmt="%.6f"
    )
    np.savetxt(
        os.path.join(out_dir, f"{comb_name}_oscr_data.txt"),
        np.vstack([fprs_sorted, ccrs_sorted]).T,
        header="fpr ccr", fmt="%.6f"
    )

    print(f"Saved: {res_path}")
    print(f"Saved: {comb_name}_roc_data.txt")
    print(f"Saved: {comb_name}_oscr_data.txt")

    return auroc, oscr_value


def plot_and_save_aggregate_roc(roc_list, loss_type, out_dir):
    plt.figure()
    grid = np.linspace(0, 1, 1000)
    interp_tprs = []
    for comb_id, fpr, tpr in roc_list:
        plt.plot(fpr, tpr, alpha=0.3, label=f"Comb {comb_id}")
        interp = np.interp(grid, fpr, tpr)
        interp[0] = 0.0
        interp_tprs.append(interp)
    mean_tpr        = np.mean(interp_tprs, axis=0)
    mean_tpr[-1]    = 1.0
    plt.plot(grid, mean_tpr, linewidth=2, color='red', label="Mean ROC")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Aggregate ROC ({loss_type})')
    plt.legend(loc='lower right')
    png_path = os.path.join(out_dir, f"{loss_type}_all_combinations_mean_ROC.png")
    txt_path = os.path.join(out_dir, f"{loss_type}_all_combinations_ROC_data.txt")
    plt.savefig(png_path)
    np.savetxt(txt_path, np.vstack([grid, mean_tpr]).T,
               header='fpr mean_tpr', fmt="%.6f")
    print(f"Saved aggregate ROC plot to {png_path}")
    print(f"Saved aggregate ROC data to {txt_path}")
    plt.close()

def plot_and_save_aggregate_oscr(oscr_list, loss_type, out_dir):
    plt.figure()
    grid = np.linspace(0, 1, 1000)
    interp_ccrs = []
    for comb_id, fprs, ccrs in oscr_list:
        plt.plot(fprs, ccrs, alpha=0.3, label=f"Comb {comb_id}")
        interp_ccrs.append(np.interp(grid, fprs, ccrs))
    mean_ccr        = np.mean(interp_ccrs, axis=0)
    plt.plot(grid, mean_ccr, linewidth=2, color='red', label="Mean OSCR")
    plt.xlabel('False Positive Rate (Unknown)')
    plt.ylabel('Closed-set Classification Rate')
    plt.title(f'Aggregate OSCR ({loss_type})')
    plt.legend(loc='lower right')
    png_path = os.path.join(out_dir, f"{loss_type}_all_combinations_mean_OSCR.png")
    txt_path = os.path.join(out_dir, f"{loss_type}_all_combinations_OSCR_data.txt")
    plt.savefig(png_path)
    np.savetxt(txt_path, np.vstack([grid, mean_ccr]).T,
               header='fpr mean_ccr', fmt="%.6f")
    print(f"Saved aggregate OSCR plot to {png_path}")
    print(f"Saved aggregate OSCR data to {txt_path}")
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Device check
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    print(f"Using device: {args.device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate each checkpoint
    all_ckpts = sorted(f for f in os.listdir(args.weights_dir) if f.endswith('.pth'))
    roc_list, oscr_list = [], []
    for ckpt in all_ckpts:
        try:
            comb_id = parse_combination_number(ckpt)
        except ValueError:
            continue
        checkpoint_path = os.path.join(args.weights_dir, ckpt)
        audio_bb_path   = os.path.join(
            args.audio_backbone_dir,
            f"openset_split_combination_{comb_id}_wav2vec_backbone.pth"
        )
        visual_bb_path  = os.path.join(
            args.visual_backbone_dir,
            f"openset_split_combination_{comb_id}_timesformer_backbone.pth"
        )
        csv_path        = os.path.join(
            args.csv_dir,
            f"multimodal-combination-{comb_id}.csv"
        )
        evaluate_single_combination_evi(
            comb_id, checkpoint_path,
            audio_bb_path, visual_bb_path,
            csv_path, args, roc_list, oscr_list
        )

    # Plot & save aggregate curves
    if roc_list and oscr_list:
        plot_and_save_aggregate_roc(roc_list, 'evi', args.output_dir)
        plot_and_save_aggregate_oscr(oscr_list, 'evi', args.output_dir)

if __name__ == '__main__':
    main()
