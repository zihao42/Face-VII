import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random


# Define RAF-DB dataset class
class RAFDBDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("L").convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Load dataset paths and labels (Modify with actual dataset location)
def load_rafdb_dataset(img_dir, label_file_path):
    train_image_paths = []
    train_labels = []
    test_image_paths = []
    test_labels = []
    with open(label_file_path, "r") as f:
        for line in f:
            img_name, label = line.strip().split()
            name, suffix = img_name.split('.')
            img_name = name + '_aligned.' + suffix
            if "train" in img_name:
                train_image_paths.append(os.path.join(img_dir, img_name))
                train_labels.append(int(label) - 1)  # Convert to zero-based index
            else:
                test_image_paths.append(os.path.join(img_dir, img_name))
                test_labels.append(int(label) - 1)  # Convert to zero-based index
    return train_image_paths, train_labels, test_image_paths, test_labels


def get_dataloaders(img_dir, label_file_path):
    train_image_paths, train_labels, test_image_paths, test_labels = load_rafdb_dataset(img_dir, label_file_path)
    combined = list(zip(train_image_paths, train_labels))
    random.shuffle(combined)
    train_image_paths, train_labels = zip(*combined)
    eval_size = int(len(train_labels) * 0.2)
    eval_image_paths, eval_labels = train_image_paths[:eval_size], train_labels[:eval_size]
    train_image_paths, train_labels = train_image_paths[eval_size:], train_labels[eval_size:]

    print("Number of train samples: " + str(len(train_image_paths)))
    print("Number of train labels: " + str(len(train_labels)))
    print("Number of eval samples: " + str(len(eval_image_paths)))
    print("Number of eval labels: " + str(len(eval_labels)))
    print("Number of test samples: " + str(len(test_image_paths)))
    print("Number of test labels: " + str(len(test_labels)))

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RAFDBDataset(train_image_paths, train_labels, transform=transform)
    eval_dataset = RAFDBDataset(eval_image_paths, eval_labels, transform=transform_eval)
    test_dataset = RAFDBDataset(test_image_paths, test_labels, transform=transform_eval)
    dataloader_train = DataLoader(train_dataset, batch_size=256, shuffle=True)
    dataloader_eval = DataLoader(eval_dataset, batch_size=256, shuffle=False)
    dataloader_test = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return dataloader_train, dataloader_eval, dataloader_test
