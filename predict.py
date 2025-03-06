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
        description="Predict image class using open-set emotion recognition model"
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights (.pth file)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image file")
    args = parser.parse_args()

    # Automatically select CUDA if available, else use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the same model configuration as in your training code.
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    num_labels = 7
    model = SwinForImageClassification.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
        num_labels=num_labels
    )
    
    # Load the saved weights.
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load and preprocess the input image.
    img_tensor = load_image(args.image).to(device)

    # Run inference.
    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs.logits  # shape: (1, 7)

    # Get the predicted index and convert to label in range 1-7.
    pred_idx = torch.argmax(logits, dim=1).item()
    predicted_class = pred_idx + 1
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
