#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from transformers import TimesformerModel


def load_timesformer_backbone(weights: str, device: torch.device) -> TimesformerModel:
    """
    加载预训练的 Timesformer backbone 模型，并使用 .pth 文件中的权重进行初始化。

    输入:
        weights: str，.pth 权重文件路径，仅包含 backbone 的 state_dict
        device: torch.device，用于模型加载

    返回:
        加载并设为 eval 模式的 TimesformerModel
    """
    # 使用已在 Hugging Face 上发布、在 Kinetics-400 上 finetune 的模型配置
    backbone = TimesformerModel.from_pretrained(
        "facebook/timesformer-base-finetuned-k400", output_hidden_states=False
    )
    state_dict = torch.load(weights, map_location=device, weights_only=True)
    backbone.load_state_dict(state_dict)
    backbone.to(device)
    backbone.eval()
    return backbone


def extract_frame_features(
    video_batch: torch.Tensor,
    combination: int,
    weights_dir: str = os.path.join("weights", "backbones", "visual"),
    device: torch.device = None
) -> torch.Tensor:
    """
    使用 Timesformer backbone 提取视频帧特征。

    输入:
        video_batch: Tensor，形状 [B, T, C, H, W]
        combination: int，对应权重文件编号 N，会加载 openset_split_combination_{N}_timesformer_backbone.pth
        weights_dir: str，权重文件所在目录
        device: torch.device 或 str，可选；若为 None，则自动选择 GPU 否则 CPU

    返回:
        frame_features: Tensor，形状 [B, T, hidden_dim]
    """
    # 选择设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    # 构造权重文件路径
    weights_filename = f"openset_split_combination_{combination}_timesformer_backbone.pth"
    weights_path = os.path.join(weights_dir, weights_filename)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"未找到权重文件: {weights_path}")

    # 加载并冻结 Timesformer 模型
    backbone = load_timesformer_backbone(weights_path, device)

    # 移动输入到设备
    video_batch = video_batch.to(device)

    # 前向推理
    outputs = backbone(video_batch)
    last_hidden = outputs.last_hidden_state  # [B, L, hidden_dim]
    B, L, hidden_dim = last_hidden.shape

    # 移除 CLS token 并按帧池化
    L_without_cls = L - 1
    T = video_batch.shape[1]
    if L_without_cls % T != 0:
        raise ValueError("Token 数量与视频帧数不匹配，无法正确进行空间平均池化。")
    num_patches = L_without_cls // T
    tokens = last_hidden[:, 1:, :].reshape(B, T, num_patches, hidden_dim)
    frame_features = tokens.mean(dim=2)  # [B, T, hidden_dim]

    return frame_features
