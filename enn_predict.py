# predict.py

import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from data import COMBINATION_SPLITS, load_video_frames, load_audio_file
from audio_feature_extract import load_audio_backbone, extract_audio_features
from visual_feature_extract import load_timesformer_backbone, extract_video_features
from enn_head import EvidentialClassificationHead
from PIL import Image

# Emotion names (original labels 0–7)
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

def load_models(combination, video_w, audio_w, clf_w, enn_w):
    """Loads multimodal backbones, fusion head, and ENN head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # backbones
    video_bb = load_timesformer_backbone(video_w, device)
    audio_bb = load_audio_backbone(audio_w, device)
    # fusion classifier
    in_dim = video_bb.config.hidden_size + audio_bb.config.hidden_size
    classifier = nn.Linear(in_dim, len(generate_label_map(combination)))
    classifier.load_state_dict(torch.load(clf_w, map_location=device))
    classifier.to(device).eval()
    # evidential head
    enn_head = EvidentialClassificationHead(in_dim, in_dim, use_bn=True)
    enn_head.load_state_dict(torch.load(enn_w, map_location=device))
    enn_head.to(device).eval()
    # label maps
    label_map = generate_label_map(combination)
    inv_map = inverse_label_map(label_map)
    return video_bb, audio_bb, classifier, enn_head, label_map, inv_map, device

def predict_one(video_path, audio_path, combination,
                video_w, audio_w, clf_w, enn_w, threshold):
    v_bb, a_bb, clf, enn, label_map, inv_map, device = load_models(
        combination, video_w, audio_w, clf_w, enn_w
    )

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # load video
    vid = load_video_frames(video_path, num_frames=32, transform=transform)
    vid = vid.unsqueeze(0).to(device)  # [1,32,C,H,W]
    # load audio
    wav = load_audio_file(audio_path)
    if wav.dim()>1: wav = wav.mean(dim=0,keepdim=True)
    wav = wav.unsqueeze(0).to(device)  # [1,1,S]

    with torch.no_grad():
        # extract features
        vf = extract_video_features(vid, v_bb).mean(dim=1)       # [1,H]
        af = extract_audio_features(wav, a_bb, target_frames=32).mean(dim=1)  # [1,H]
        fused = torch.cat([vf, af], dim=1)                       # [1,2H]

        # classification logits (not used for unknown_score)
        logits = clf(fused)                                      # [1,K]
        # evidential
        evidence = enn(fused)                                    # [1,K]
        alpha = evidence + 1.0                                   # Dirichlet α
        S = alpha.sum(dim=1)                                     # [1]
        vacuity = (alpha.shape[1] / S).item()                    # scalar

        # soft Dirichlet mean probs
        probs = alpha / alpha.sum(dim=1, keepdim=True)           # [1,K]
        maxp, idx = probs.max(dim=1)

    # decide
    if vacuity > threshold:
        print(f"Predicted: Unknown (vacuity={vacuity:.4f})")
    else:
        orig = inv_map[idx.item()]
        name = EMOTION_NAMES.get(orig, "Unknown")
        print(f"Predicted: {orig} ({name}), vacuity={vacuity:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Multimodal Open-Set Predict")
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--combination", type=int, required=True, choices=list(COMBINATION_SPLITS.keys()))
    parser.add_argument("--video_weights", required=True)
    parser.add_argument("--audio_weights", required=True)
    parser.add_argument("--classifier_weights", required=True)
    parser.add_argument("--enn_weights", required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Vacuity threshold (0–1) for unknown detection")
    args = parser.parse_args()
    predict_one(
        args.video, args.audio, args.combination,
        args.video_weights, args.audio_weights,
        args.classifier_weights, args.enn_weights,
        args.threshold
    )
