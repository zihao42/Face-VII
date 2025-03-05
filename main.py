import torch
import random
from train import train
from data import get_dataloaders


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(42)
    img_dir = "./RAF-DB/aligned"
    label_file_path = "./RAF-DB/list_patition_label.txt"

    dataloader_train, dataloader_eval, dataloader_test = get_dataloaders(img_dir, label_file_path)

    weights_dir = "./weights"
    train(10, 3, 7, dataloader_train, dataloader_eval, 5, weights_dir, evi=True)
    # to enable batch_normalization, set use_bn=True
    # train(10, 3, 7, dataloader_train, dataloader_eval, 5, weights_dir, evi=True, use_bn=True)
