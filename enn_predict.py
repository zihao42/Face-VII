# predict.py

import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from data import COMBINATION_SPLITS, load_video_frames, load_audio_file
from audio_feature_extract import load_audio_backbone, extract_audio_features
from visual_feature_extract import load_timesformer_backbone, extract_video_features
from fusion_model import MultimodalTransformer
from enn_head import EvidentialClassificationHead

# Emotion annotation mapping
EMOTION_NAMES = {
    0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad",
    4: "Angry", 5: "Fearful", 6: "Disgust", 7: "Surprise"
}


def generate_label_map(combination: int):
    combo = COMBINATION_SPLITS[combination]
    known = sorted(combo['known'])
    return {orig: idx for idx, orig in enumerate(known)}


def inverse_label_map(label_map):
    return {v: k for k, v in label_map.items()}


def load_models(combination, video_w, audio_w, fusion_w,
                input_dim=768, embed_dim=128, num_heads=8, num_layers=1):
    """Load video/audio backbones, fusion transformer, and ENN head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Backbones
    video_bb = load_timesformer_backbone(video_w, device)
    audio_bb = load_audio_backbone(audio_w, device)
    # Fusion transformer (with classifier)
    num_known = len(generate_label_map(combination))
    fusion = MultimodalTransformer(
        modality_num=2,
        num_classes=num_known,
        feature_only=False,
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    fusion.load_state_dict(torch.load(fusion_w, map_location=device))
    fusion.to(device).eval()
    # Evidential head on fused features
    enn_head = EvidentialClassificationHead(embed_dim * 2, num_known, use_bn=True)
    enn_head.load_state_dict(torch.load(enn_w, map_location=device))
    enn_head.to(device).eval()
    # Label maps
    label_map = generate_label_map(combination)
    inv_map = inverse_label_map(label_map)
    return video_bb, audio_bb, fusion, label_map, inv_map, device


def predict_batch(vids, auds, models, threshold=0.5):
    """
    Batch inference: returns lists of preds, vacuity scores, max_probs.
    vids: Tensor [B, T, C, H, W]
    auds: Tensor [B, ...] raw waveform
    models: tuple(video_bb, audio_bb, fusion, enn_head, label_map, inv_map, device)
    """
    video_bb, audio_bb, fusion, label_map, inv_map, device = models
    B = vids.size(0)
    with torch.no_grad():
        # features
        v_feats = extract_video_features(vids.to(device), video_bb).mean(dim=1)
        a_feats = extract_audio_features(auds.to(device), audio_bb, target_frames=v_feats.shape[1]).mean(dim=1)
        # fusion
        logits, fused_feats = fusion([v_feats.unsqueeze(1), a_feats.unsqueeze(1)])
        # evidential
        evidence = enn_head(fused_feats)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1)
        vacuity = (alpha.shape[1] / S).tolist()
        probs = alpha / alpha.sum(dim=1, keepdim=True)
        maxp, idx = probs.max(dim=1)
    preds = []
    for i in range(B):
        if vacuity[i] > threshold:
            preds.append(8)
        else:
            orig = inv_map[idx[i].item()]
            preds.append(orig)
    return preds, vacuity, maxp.tolist()


def predict_one(video_path, audio_path, combination,
                video_w, audio_w, fusion_w, enn_w,
                threshold):
    models = load_models(combination, video_w, audio_w, fusion_w, enn_w)
    # preprocess same as batch but single
    vid = load_video_frames(video_path, num_frames=32,
                            transform=transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                            ])).unsqueeze(0)
    wav = load_audio_file(audio_path)
    if wav.dim()>1: wav=wav.mean(dim=0,keepdim=True)
    wav = wav.unsqueeze(0)
    preds, vacs, maxps = predict_batch(vid, wav, models, threshold)
    p, v = preds[0], vacs[0]
    if p==8:
        print(f"Predicted: Unknown (vacuity={v:.4f})")
    else:
        name = EMOTION_NAMES.get(p, "Unknown")
        print(f"Predicted: {p} ({name}), vacuity={v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--combination", type=int, required=True, choices=list(COMBINATION_SPLITS.keys()))
    parser.add_argument("--video_weights", required=True)
    parser.add_argument("--audio_weights", required=True)
    parser.add_argument("--fusion_weights", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    predict_one(
        args.video, args.audio, args.combination,
        args.video_weights, args.audio_weights,
        args.fusion_weights, 
        args.threshold
    )


