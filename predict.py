import argparse
import torch
from transformers import SwinForImageClassification
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_path):
    """
    Loads an image from the given file path and applies evaluation preprocessing:
      - Resize to 224x224,
      - Convert to tensor,
      - Normalize using ImageNet statistics.
    """
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform_eval(image).unsqueeze(0)  # add batch dimension

def main():
    parser = argparse.ArgumentParser(
        description="Predict image class with unknown detection using open-set emotion recognition model"
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights (.pth file)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image file")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Softmax probability threshold for unknown detection (default: 0.65)")
    args = parser.parse_args()

    # Automatically select CUDA if available, else use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model with 6 labels (matching your training configuration).
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    num_labels = 6
    model = SwinForImageClassification.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
        num_labels=num_labels
    )
    
    # Load the saved weights (assumed to be for 6 labels).
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load and preprocess the input image.
    img_tensor = load_image(args.image).to(device)

    # Run inference.
    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs.logits  # shape: (1, 6)
        probs = torch.softmax(logits, dim=1)
        max_prob, pred_idx = torch.max(probs, dim=1)
    
    # Check threshold to decide if sample is known or unknown.
    if max_prob.item() < args.threshold:
        print("Predicted class: Unknown")
    else:
        annot = ""
        # Convert zero-indexed prediction to label (1-6)
        predicted_class = pred_idx.item() + 1
        if predicted_class == 1:
            annot= "Surprised"
        elif predicted_class == 2:
            annot = "Fear"
        elif predicted_class == 3:
            annot = "Disgust"
        elif predicted_class == 4:
            annot = "Happiness"
        elif predicted_class == 5:
            annot = "Sadness"
        elif predicted_class == 6:
            annot = "Anger"
        elif predicted_class == 7:
            annot = "Neutral"

        print(f"Predicted class: {predicted_class} {annot} (confidence: {max_prob.item():.3f})")

if __name__ == "__main__":
    main()
