import os
import re
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from fusion_train import RAVDESSMultimodalDataset, collate_fn_modality
from feature_fusion import MultimodalTransformer
from audio_feature_extract import load_audio_backbone, extract_audio_features_from_backbone
from visual_feature_extract import load_timesformer_backbone, extract_frame_features_from_backbone
from enn_head import EvidentialClassificationHead


def parse_combination_number(filename):
    m = re.search(r'combination[-_]?(\d+)', filename)
    if not m:
        raise ValueError(f"Cannot parse combination number from {filename}")
    return int(m.group(1))


def evaluate_single_combination_evi(
    comb_id, clf_path,
    audio_bb_path, visual_bb_path,
    csv_path, args
):
    print(f"\n--- Combination {comb_id} (EVI close-set) ---")
    print(f"  Classifier:      {clf_path}")
    print(f"  Audio backbone:  {audio_bb_path}")
    print(f"  Visual backbone: {visual_bb_path}")
    print(f"  CSV file:        {csv_path}")

    # build label mapping train ∪ test_known
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

    # prepare test samples, just use val_df here
    samples = list(zip(
        val_df.audio_filename.values,
        val_df.video_filename.values,
        val_df.emo_label.values
    ))

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

    # load backbones
    device = torch.device(args.device)
    backbone_v = load_timesformer_backbone(visual_bb_path, device)
    backbone_a = load_audio_backbone(audio_bb_path, device)

    # load evi model
    raw_state = torch.load(clf_path, map_location=device)
    model = MultimodalTransformer(
        modality_num=2,
        num_classes=num_classes,
        num_layers=2,
        feature_only=True
    ).to(device)
    model.load_state_dict(raw_state['model'])
    enn_head = EvidentialClassificationHead(
        model.embed_dim * model.n_modality,
        num_classes,
        use_bn=True
    ).to(device)
    enn_head.load_state_dict(raw_state['enn_head'])
    model.eval()
    enn_head.eval()

    # inference & collect
    y_true_idx, y_pred_idx = [], []
    with torch.no_grad():
        for wavs, vids, lbls in tqdm(loader, desc=f"Comb {comb_id}", leave=False):
            wavs, vids = wavs.to(device), vids.to(device)
            feat_v = extract_frame_features_from_backbone(vids, backbone_v)
            feat_a = extract_audio_features_from_backbone(wavs, backbone_a)
            fused  = model([feat_a, feat_v])
            evidence = enn_head(fused)
            alpha    = evidence + 1.0
            probs    = alpha / alpha.sum(dim=1, keepdim=True)
            preds    = probs.argmax(dim=1).cpu().numpy()
            for raw_lbl, p in zip(lbls, preds):
                y_true_idx.append(int(raw_lbl.item()))
                y_pred_idx.append(int(p))

    y_true_idx = np.array(y_true_idx, dtype=int)
    y_pred_idx = np.array(y_pred_idx, dtype=int)

    # confusion matrix
    cm = confusion_matrix(
        y_true_idx,
        y_pred_idx,
        labels=list(range(num_classes))
    )
    # save CSV
    pd.DataFrame(cm, index=unique_lbls, columns=unique_lbls).to_csv(os.path.join(args.output_dir, f"combination-{comb_id}_confusion_evi.csv"),
              index_label='true')
    return cm, unique_lbls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir',        type=str, required=True,
                        help='EVI model weights directory')
    parser.add_argument('--audio_backbone_dir',  type=str, required=True)
    parser.add_argument('--visual_backbone_dir', type=str, required=True)
    parser.add_argument('--csv_dir',             type=str, required=True)
    parser.add_argument('--media_dir',           type=str, required=True)
    parser.add_argument('--batch_size',          type=int, default=32)
    parser.add_argument('--num_frames',          type=int, default=32)
    parser.add_argument('--device',              type=str, default='cuda')
    parser.add_argument('--output_dir',          type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'

    # iterate all evi checkpoints
    all_clfs = sorted(f for f in os.listdir(args.weights_dir) if f.endswith('.pth'))
    cms, metrics_list = [], []
    full_labels = list(range(8))

    for clf in all_clfs:
        try:
            comb_id = parse_combination_number(clf)
        except ValueError:
            continue
        clf_path      = os.path.join(args.weights_dir, clf)
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
        cm, labels = evaluate_single_combination_evi(
            comb_id, clf_path,
            audio_bb_path, visual_bb_path,
            csv_path, args
        )
        # zero-pad to 8x8
        cm8 = np.zeros((8,8), dtype=cm.dtype)
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                cm8[li, lj] = cm[i,j]
        cms.append(cm8)
        # compute metrics
        total    = cm.trace()
        total_n  = cm.sum()
        accuracy = total / total_n if total_n>0 else 0.0
        precisions = []
        recalls    = []
        for k in range(len(labels)):
            tp = cm[k,k]
            fp = cm[:,k].sum() - tp
            fn = cm[k,:].sum() - tp
            precisions.append(tp/(tp+fp) if (tp+fp)>0 else 0.0)
            recalls.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)
        metrics_list.append({
            'combination': comb_id,
            'accuracy': accuracy,
            'precision': np.mean(precisions),
            'recall':    np.mean(recalls)
        })

    # save average confusion
    df_avg = pd.DataFrame(sum(cms)/len(cms), index=full_labels, columns=full_labels)
    df_avg.to_csv(os.path.join(args.output_dir, "average_confusion_evi.csv"), index_label='label')
    plt.figure(figsize=(6,6))
    plt.imshow(df_avg.values, cmap='Blues', vmin=0, vmax=df_avg.values.max())
    plt.title("Average Confusion Matrix (EVI)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "average_confusion_evi.png"))
    plt.close()

    # save metrics summary
    df_metrics = pd.DataFrame(metrics_list).set_index('combination')
    df_metrics.loc['average'] = df_metrics.mean()
    df_metrics.to_csv(os.path.join(args.output_dir, "metrics_summary_evi.csv"), index_label='combination')

    print(f"All EVI close-set evaluation results saved in {args.output_dir}")


if __name__ == "__main__":
    main()
