import torch
import random 
import argparse
from train import train
from data import get_dataloaders


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", nargs=2, default=(70, "default_str"), help="train mode")
    args = parser.parse_args()
    uk_mode, uk = args.train_mode

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(42)
    img_dir = "/media/data1/ningtong/wzh/projects/data/Image/aligned"
    label_file_path = "/media/data1/ningtong/wzh/projects/data/Image/list_patition_label.txt"

    dataloader_train, dataloader_eval, dataloader_test = get_dataloaders(img_dir, label_file_path, uk_mode, uk)

    weights_dir = "/media/data1/ningtong/wzh/projects/Face-VII/weights"
    
    
    train(num_epochs=15, eval_gap_epoch=1, num_labels=7, dataloader_train=dataloader_train,
          dataloader_eval=dataloader_eval, save_weights_gap_epoch=5, save_weight_dir=weights_dir,
          use_variance=True, use_schedule=True)
