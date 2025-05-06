import os
import torch
from tqdm import tqdm
import timm
import numpy as np
import argparse
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from data import get_openset_dataloaders, COMBINATION_SPLITS

def generate_label_map(combination_id):

    if combination_id not in COMBINATION_SPLITS:
        raise ValueError(f"Combination {combination_id} is not defined. Available combinations: {list(COMBINATION_SPLITS.keys())}")
    known_labels = COMBINATION_SPLITS[combination_id]["known"]
    label_map = {orig: new for new, orig in enumerate(sorted(known_labels))}
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map

def map_labels(labels, mapping):

    mapped = torch.tensor([mapping[int(x)] for x in labels.cpu().tolist()],
                            dtype=torch.long, device=labels.device)
    return mapped

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map, inv_label_map = generate_label_map(args.combination)
    print(f"当前组合编号：{args.combination}")
    print(f"Known label mapping (original -> mapped): {label_map}")
    print(f"Inverse mapping (mapped -> original): {inv_label_map}")

    video_dir = "/path/to/ravdess/videos"
    audio_dir = "/path/to/ravdess/audios"
    label_file = "/path/to/labels.xlsx"

    train_loader, _, _ = get_openset_dataloaders(
        video_dir,
        audio_dir,
        label_file,
        modality='both',
        batch_size=4,
        num_frames=16,
        test_ratio=0.1,
        eval_ratio=0.2
    )

    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    audio_model.to(device)
    audio_model.eval()

    video_model = timm.create_model('swin_base_patch244_kinetics400', pretrained=True)

    if hasattr(video_model, 'global_pool'):
        video_model.global_pool = torch.nn.Identity()
    video_model.to(device)
    video_model.eval()

    output_dir = "/media/data1/ningtong/wzh/projects/Face-VII/features"
    os.makedirs(output_dir, exist_ok=True)

    sample_index = 0

    for data, label, info in tqdm(train_loader, desc="Extracting features"):

        audio_waveform = data['audio']

        if audio_waveform.shape[1] > 1:
            audio_waveform = torch.mean(audio_waveform, dim=1)
        else:
            audio_waveform = audio_waveform.squeeze(1)

        audio_features_list = []
        for waveform in audio_waveform:

            inputs = audio_processor(waveform.cpu().numpy(), sampling_rate=16000, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = audio_model(**inputs)

            feat = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            audio_features_list.append(feat.cpu())
        audio_features = torch.stack(audio_features_list)

        video_tensor = data['video'].permute(0, 2, 1, 3, 4).to(device)
        with torch.no_grad():
            video_feats = video_model.forward_features(video_tensor)
            video_features = video_feats

        mapped_labels = map_labels(label, label_map)

        inversed_labels = torch.tensor(
            [inv_label_map[int(x)] for x in mapped_labels.cpu().tolist()],
            dtype=torch.long,
            device=mapped_labels.device
        )

        batch_size = audio_features.shape[0]
        for i in range(batch_size):
            feature_dict = {
                "audio_feature": audio_features[i],
                "video_feature": video_features[i],
                "label": inversed_labels[i]
            }
            save_path = os.path.join(output_dir, f"sample_{sample_index}.pt")
            torch.save(feature_dict, save_path)
            sample_index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于组合划分进行标签映射及逆映射的特征提取器")
    parser.add_argument('--combination', type=int, required=True,
                        help="用于标签映射的组合编号（1-10）")
    args = parser.parse_args()
    main(args)
