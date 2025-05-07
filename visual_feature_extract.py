import os
import torch
from transformers import TimesformerModel

def load_timesformer_backbone(weights: str, device: torch.device) -> TimesformerModel:
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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    fname = f"openset_split_combination_{combination}_timesformer_backbone.pth"
    wpath = os.path.join(weights_dir, fname)
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Cannot find weights files: {wpath}")

    backbone = load_timesformer_backbone(wpath, device)
    video_batch = video_batch.to(device)

    outputs = backbone(video_batch)
    last_hidden = outputs.last_hidden_state
    B, L, D = last_hidden.shape

    Lw = L - 1
    T = video_batch.shape[1]
    if Lw % T != 0:
        raise ValueError("token does not match with frame rates")
    P = Lw // T

    tokens = last_hidden[:, 1:, :].reshape(B, T, P, D)
    frame_features = tokens.mean(dim=2)
    return frame_features

def extract_frame_features_from_backbone(
    video_batch: torch.Tensor,
    backbone: TimesformerModel
) -> torch.Tensor:
    device = next(backbone.parameters()).device
    video_batch = video_batch.to(device)

    outputs = backbone(video_batch)
    last_hidden = outputs.last_hidden_state  # [B, L, D]
    B, L, D = last_hidden.shape

    Lw = L - 1
    T = video_batch.shape[1]
    if Lw % T != 0:
        raise ValueError("token does not match with frame rates")
    P = Lw // T

    tokens = last_hidden[:, 1:, :].reshape(B, T, P, D)
    frame_features = tokens.mean(dim=2)  # [B, T, D]
    return frame_features
