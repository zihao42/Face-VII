# predict.py
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from data import COMBINATION_SPLITS, load_video_frames, load_audio_file
from audio_feature_extract import load_audio_backbone, extract_audio_features
from visual_feature_extract import load_timesformer_backbone, extract_video_features
from PIL import Image

# Emotion annotation mapping
EMOTION_NAMES = {
    0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad",
    4: "Angry", 5: "Fearful", 6: "Disgust", 7: "Surprise"
}

def generate_label_map(combination: int):
    combo = COMBINATION_SPLITS[combination]
    known = sorted(combo['known'])
    return {orig: idx for idx, orig in enumerate(known)}

def inverse_label_map(label_map: dict):
    return {v: k for k, v in label_map.items()}

def load_models(combination: int, video_weights: str,
                audio_weights: str, classifier_weights: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_bb = load_timesformer_backbone(video_weights, device)
    audio_bb = load_audio_backbone(audio_weights, device)
    label_map = generate_label_map(combination)
    inv_map = inverse_label_map(label_map)
    num_known = len(label_map)
    in_dim = video_bb.config.hidden_size + audio_bb.config.hidden_size
    classifier = nn.Linear(in_dim, num_known)
    classifier.load_state_dict(torch.load(classifier_weights, map_location=device))
    classifier.to(device).eval()
    return video_bb, audio_bb, classifier, label_map, inv_map, device

def predict_batch(videos, audios,
                  video_bb, audio_bb, classifier,
                  label_map, inv_map,
                  threshold: float):
    device = next(classifier.parameters()).device
    videos = videos.to(device)
    B = videos.size(0)
    # prepare audio
    if audios.dim() == 3 and audios.size(1) != 1:
        audios = audios.view(B, -1).unsqueeze(1)
    audios = audios.to(device)
    with torch.no_grad():
        v_feats = extract_video_features(videos, video_bb)
        v_pooled = v_feats.mean(dim=1)
        a_feats = extract_audio_features(audios, audio_bb, target_frames=v_feats.shape[1])
        a_pooled = a_feats.mean(dim=1)
        fused = torch.cat([v_pooled, a_pooled], dim=1)
        logits = classifier(fused)
        probs = torch.softmax(logits, dim=1)
        max_prob, idx = probs.max(dim=1)
    preds, unk_scores = [], []
    for i in range(B):
        if max_prob[i].item() < threshold:
            preds.append(8)
        else:
            preds.append(inv_map[idx[i].item()])
        unk_scores.append(1 - max_prob[i].item())
    return preds, unk_scores, max_prob.cpu()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--combination", type=int,
                        choices=list(COMBINATION_SPLITS.keys()), required=True)
    parser.add_argument("--video_weights", required=True)
    parser.add_argument("--audio_weights", required=True)
    parser.add_argument("--classifier_weights", required=True)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Probability threshold for unknown detection")
    args = parser.parse_args()
    video_bb, audio_bb, classifier, label_map, inv_map, device = load_models(
        args.combination,
        args.video_weights,
        args.audio_weights,
        args.classifier_weights
    )
    # transforms
    video_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    vid = load_video_frames(args.video, num_frames=32,
                            transform=video_transform).unsqueeze(0)
    wav = load_audio_file(args.audio)
    if wav.dim() > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0)
    preds, scores, _ = predict_batch(
        vid, wav,
        video_bb, audio_bb, classifier,
        label_map, inv_map,
        args.threshold
    )
    p = preds[0]
    s = scores[0]
    if p == 8:
        print(f"Predicted: Unknown, score={s:.4f}")
    else:
        name = EMOTION_NAMES.get(p, "Unknown")
        print(f"Predicted: {p} ({name}), score={s:.4f}")

if __name__ == "__main__":
    main()