import argparse
import os
import torch
from transformers import SwinForImageClassification, logging as hf_logging
import torchvision.transforms as transforms
from PIL import Image
import warnings
from enn_head import EvidentialClassificationHead

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

def generate_label_map(uk):
    """
    根据 uk（unknown key）生成正向标签映射字典。
    例如，如果 uk 为 "sur"，则会舍弃标签 0，其余标签 [1,2,3,4,5,6] 会映射为 0~5。
    """
    if uk == "sur":
        chosen_label = 0
    elif uk == "fea":
        chosen_label = 1
    elif uk == "dis":
        chosen_label = 2
    elif uk == "hap":
        chosen_label = 3
    elif uk == "sad":
        chosen_label = 4
    elif uk == "ang":
        chosen_label = 5
    elif uk == "neu":
        chosen_label = 6
    else:
        chosen_label = None
    
    valid_labels = [i for i in range(7) if i != chosen_label]
    label_map = {orig: new for new, orig in enumerate(sorted(valid_labels))}
    return label_map

def extract_uk_from_weights(weights_path):
    """
    根据权重文件名，提取 uk（例如 "sur", "fea", ...）。
    这里通过判断文件名中是否包含特定子字符串来实现。
    """
    uk_options = ["sur", "fea", "dis", "hap", "sad", "ang", "neu"]
    for uk in uk_options:
        if uk in os.path.basename(weights_path):
            return uk
    return None

def load_image(image_input):
    """
    Loads an image and applies evaluation preprocessing.
    
    If the input is a tensor, it is assumed to be already preprocessed;
    if it does not have a batch dimension, one is added.
    
    Otherwise, the image is loaded from the file path, resized to 224x224,
    converted to a tensor, normalized using ImageNet statistics, and a batch
    dimension is added.
    """
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image_input, torch.Tensor):
        if image_input.ndim == 3:
            return image_input.unsqueeze(0)
        return image_input
    else:
        image = Image.open(image_input).convert("RGB")
        return transform_eval(image).unsqueeze(0)

def predict_image(weights, image, threshold=0.7, model=None, enn_head=None):
    """
    根据给定的 weights 加载模型，处理 image（可以是文件路径或 tensor），
    返回预测类别及其注释信息：
      - 若最大 softmax 概率低于 threshold，则预测为未知（类别 8）；
      - 否则，将模型输出（经过逆映射后）转换为 1-indexed 的原始标签，并返回相应的情绪注释。

    如果传入了预加载的 model（非 None），则直接使用该 model 进行预测，
    从而避免重复加载模型，并确保输入 tensor 被移动到 model 所在设备上。
    """
    # 如果传入了预加载 model，则使用 model 所在设备，否则加载模型
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "microsoft/swin-tiny-patch4-window7-224"
        num_labels = 6
        model = SwinForImageClassification.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
            num_labels=num_labels
        )
        state_dict = torch.load(weights, map_location=device)
        # also load enn head if available
        if 'evi_head_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
            # now default use_bn=True, will revise in later versions
            enn_head = EvidentialClassificationHead(model.config.hidden_size, num_labels, use_bn=True) 
            enn_head.load_state_dict(state_dict['evi_head_state_dict'])
            enn_head.to(device)
            enn_head.eval()
        else: 
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    else:
        device = next(model.parameters()).device

    # 获取图像 tensor，并移动到 model 所在设备
    img_tensor = load_image(image).to(device)
    
    with torch.no_grad():
        if enn_head is not None:
            # hook the features like we did in train.py
            pooled_representation = {}
            def hook_fn(module, input, output):
                pooled_representation["features"] = output.pooler_output
            model.swin.register_forward_hook(hook_fn)
            # for the hook only
            _ = model(img_tensor)
            features = pooled_representation["features"]
            evidence = enn_head(features)
            # convert evidence to Dirichlet parameters
            alpha = evidence + 1.0
            # compute prob with D distribution
            probs = alpha / alpha.sum(dim=1, keepdim=True)
            max_prob, pred_idx = torch.max(probs, dim=1)
        else:
            outputs = model(img_tensor)
            logits = outputs.logits  # shape: (1,6)
            probs = torch.softmax(logits, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
    
    # 根据阈值判断：若最大概率低于 threshold，则视为未知（类别 8）
    if max_prob.item() < threshold:
        predicted_class = 8
        annot = "Unknown"
    else:
        # 尝试通过权重文件名提取 uk，并进行逆映射
        uk = extract_uk_from_weights(weights)
        if uk is not None:
            forward_map = generate_label_map(uk)
            # 生成逆映射字典：mapped -> original
            inverse_map = {v: k for k, v in forward_map.items()}
            # pred_idx 是模型预测的 mapped 索引（0-indexed）
            original_label = inverse_map.get(pred_idx.item(), None)
            if original_label is not None:
                # 为与 evaluation 中 ground truth 保持一致，输出 1-indexed 标签
                predicted_class = original_label + 1
            else:
                predicted_class = pred_idx.item() + 1
        else:
            predicted_class = pred_idx.item() + 1

        # 根据原始标签确定情绪注释
        annot_dict = {
            1: "Surprised",
            2: "Fear",
            3: "Disgust",
            4: "Happiness",
            5: "Sadness",
            6: "Anger",
            7: "Neutral"
        }
        annot = annot_dict.get(predicted_class, "Unknown")
    
    return predicted_class, annot

def main():
    parser = argparse.ArgumentParser(
        description="Predict image class with unknown detection using open-set emotion recognition model"
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights (.pth file)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image file or an image tensor")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Softmax probability threshold for unknown detection (default: 0.7)")
    args = parser.parse_args()

    predicted_class, annot = predict_image(args.weights, args.image, args.threshold)
    print(f"Predicted class: {predicted_class} {annot}")

if __name__ == "__main__":
    main()
