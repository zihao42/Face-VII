#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from tqdm import tqdm
import timm
import numpy as np
import argparse

# 从 transformers 导入 Wav2Vec2 模型及其特征提取器
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# 从 data.py 中导入数据加载函数以及组合划分的定义
from data import get_openset_dataloaders, COMBINATION_SPLITS

def generate_label_map(combination_id):
    """
    根据传入的组合编号（1～10）生成标签映射字典和逆映射字典。
    
    参数：
      combination_id: 组合编号（整数），例如 1、2、…、10
      
    返回：
      label_map: 字典，将 known 标签映射为连续索引，例如 {原始标签: 新标签}。
      inv_label_map: 逆映射字典，例如 {新标签: 原始标签}。
    """
    if combination_id not in COMBINATION_SPLITS:
        raise ValueError(f"Combination {combination_id} is not defined. Available combinations: {list(COMBINATION_SPLITS.keys())}")
    known_labels = COMBINATION_SPLITS[combination_id]["known"]
    # 按照顺序生成连续的索引映射（例如：{1: 0, 4: 1, 5: 2, 6: 3, 7: 4}）
    label_map = {orig: new for new, orig in enumerate(sorted(known_labels))}
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map

def map_labels(labels, mapping):
    """
    根据 mapping 字典，将标签 tensor 中的每个值转换为连续的标签（类型为 torch.long）。
    """
    mapped = torch.tensor([mapping[int(x)] for x in labels.cpu().tolist()],
                            dtype=torch.long, device=labels.device)
    return mapped

def main(args):
    # 设置设备（优先使用 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根据传入的组合编号生成标签映射和逆映射
    label_map, inv_label_map = generate_label_map(args.combination)
    print(f"当前组合编号：{args.combination}")
    print(f"Known label mapping (original -> mapped): {label_map}")
    print(f"Inverse mapping (mapped -> original): {inv_label_map}")

    # ===================== 数据路径设置 =====================
    video_dir = "/path/to/ravdess/videos"         # 视频文件所在目录
    audio_dir = "/path/to/ravdess/audios"          # 音频文件所在目录（若音频嵌入视频中，可设为 None）
    label_file = "/path/to/labels.xlsx"            # 标签文件路径

    # ===================== 加载训练数据 =====================
    # 此处假设使用 data.py 中定义的 get_dataloaders（可根据实际情况调整）
    train_loader, _, _ = get_openset_dataloaders(
        video_dir,
        audio_dir,
        label_file,
        modality='both',
        batch_size=4,
        num_frames=16,     # 采样视频帧数
        test_ratio=0.1,
        eval_ratio=0.2
    )

    # ===================== 初始化 Wav2Vec 2.0 模型 =====================
    # 使用 Hugging Face 上的 facebook/wav2vec2-base 模型
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    audio_model.to(device)
    audio_model.eval()

    # ===================== 初始化 Swin Transformer 3D 模型 =====================
    # 使用 timm 库加载预训练的 swin_base_patch244_kinetics400 模型
    video_model = timm.create_model('swin_base_patch244_kinetics400', pretrained=True)
    # 为了保留时序信息，将全局池化层替换为 Identity
    if hasattr(video_model, 'global_pool'):
        video_model.global_pool = torch.nn.Identity()
    video_model.to(device)
    video_model.eval()

    # ===================== 设置特征保存目录 =====================
    output_dir = "/media/data1/ningtong/wzh/projects/Face-VII/features"
    os.makedirs(output_dir, exist_ok=True)

    sample_index = 0  # 用于保存文件时的编号

    # ===================== 遍历训练数据，提取特征 =====================
    for data, label, info in tqdm(train_loader, desc="Extracting features"):
        # --------------------- 音频特征提取 ---------------------
        audio_waveform = data['audio']  # 形状 (batch, channels, time)
        # 如果存在多个通道，则取均值；否则 squeeze 去掉 channel 维度
        if audio_waveform.shape[1] > 1:
            audio_waveform = torch.mean(audio_waveform, dim=1)
        else:
            audio_waveform = audio_waveform.squeeze(1)  # shape: (batch, time)

        audio_features_list = []
        for waveform in audio_waveform:
            # 使用 Wav2Vec2 特征提取器预处理音频，转换为模型输入格式
            inputs = audio_processor(waveform.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
            # 将输入移至设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = audio_model(**inputs)
            # outputs.last_hidden_state 的形状为 (1, sequence_length, 768)
            # 对时间维度进行平均池化，得到 768 维向量（音频全局特征）
            feat = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            audio_features_list.append(feat.cpu())
        # 最终音频特征的形状为 (batch, 768)
        audio_features = torch.stack(audio_features_list)

        # --------------------- 视频特征提取 ---------------------
        # 原始形状为 (batch, num_frames, C, H, W)，转换为 (batch, C, T, H, W)
        video_tensor = data['video'].permute(0, 2, 1, 3, 4).to(device)
        with torch.no_grad():
            video_feats = video_model.forward_features(video_tensor)
            video_features = video_feats

        # --------------------- 标签映射及逆映射 ---------------------
        # 先将原始标签映射为连续数值（仅对 known 标签生效），再通过逆映射恢复到原始 known 标签
        mapped_labels = map_labels(label, label_map)
        # 对 mapped_labels 进行逆映射（恢复成原始的 known 标签）
        inversed_labels = torch.tensor(
            [inv_label_map[int(x)] for x in mapped_labels.cpu().tolist()],
            dtype=torch.long,
            device=mapped_labels.device
        )

        # --------------------- 保存特征 ---------------------
        # 此处只保存经过逆映射后的 label
        batch_size = audio_features.shape[0]
        for i in range(batch_size):
            feature_dict = {
                "audio_feature": audio_features[i],   # 768 维音频特征
                "video_feature": video_features[i],     # 视频中间层特征（包含时序信息）
                "label": inversed_labels[i]             # 经过逆映射后的 label，用于分类
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
