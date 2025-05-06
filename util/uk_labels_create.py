import os
import shutil


def create_dataset_structure(base_folder):
    dataset_path = os.path.join(base_folder, "dataset")
    data_path = os.path.join(dataset_path, "data")
    os.makedirs(data_path, exist_ok=True)
    
    # Create subfolders 1 through 7 under dataset/data
    for label in range(1, 8):
        label_folder = os.path.join(data_path, str(label))
        os.makedirs(label_folder, exist_ok=True)
    
    print(f"Created image dataset structure under: {data_path}")
    return data_path


def copy_files_according_to_label(base_folder, data_path):
    aligned_folder = os.path.join(base_folder, "aligned")
    label_file = os.path.join(base_folder, "list_patition_label.txt")
    
    if not os.path.exists(aligned_folder):
        print(f"Error: 'aligned' folder not found in {base_folder}")
        return
    if not os.path.isfile(label_file):
        print(f"Error: 'list_patition_label.txt' not found in {base_folder}")
        return

    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            parts = line.split()
            if len(parts) != 2:
                print(f"Skipping line (unexpected format): {line}")
                continue
            file_name, lbl = parts

            if lbl not in [str(i) for i in range(1, 8)]:
                print(f"Skipping file {file_name}: label {lbl} is not in 1-7")
                continue

            if file_name.endswith(".jpg"):
                aligned_file_name = file_name.replace(".jpg", "_aligned.jpg")
            else:
                aligned_file_name = file_name

            src_path = os.path.join(aligned_folder, aligned_file_name)
            dst_path = os.path.join(data_path, lbl, file_name)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")
            else:
                print(f"Source file not found: {src_path}")


def create_label_files(base_folder):
    dataset_path = os.path.join(base_folder, "dataset")
    labels_folder = os.path.join(dataset_path, "labels")
    ku61_folder = os.path.join(labels_folder, "KU61")
    os.makedirs(ku61_folder, exist_ok=True)

    uk_folders = {}
    for i in range(1, 8):
        folder = os.path.join(ku61_folder, f"UK{i}")
        os.makedirs(folder, exist_ok=True)
        uk_folders[str(i)] = folder

    variant_train_lines = {str(i): [] for i in range(1, 8)}
    variant_test_lines  = {str(i): [] for i in range(1, 8)}
    
    label_file = os.path.join(base_folder, "list_patition_label.txt")
    if not os.path.isfile(label_file):
         print(f"Error: 'list_patition_label.txt' not found in {base_folder}")
         return
    
    with open(label_file, "r") as f:
        for line in f:
            orig_line = line.strip()
            if not orig_line:
                continue
            parts = orig_line.split()
            if len(parts) != 2:
                print(f"Skipping line with unexpected format: {orig_line}")
                continue
            file_name, lbl = parts

            lower_name = file_name.lower()
            if "train" in lower_name:
                split = "train"
            elif "test" in lower_name:
                split = "test"
            else:
                print(f"Skipping line because split cannot be determined: {orig_line}")
                continue

            for i in range(1, 8):
                variant = str(i)
                if split == "train":

                    if lbl != variant:
                        variant_train_lines[variant].append(orig_line)
                elif split == "test":

                    if lbl == variant:
                        new_line = " ".join([file_name, "8"])
                        variant_test_lines[variant].append(new_line)
                    else:
                        variant_test_lines[variant].append(orig_line)

    for variant, folder in uk_folders.items():
        train_path = os.path.join(folder, "train.txt")
        test_path = os.path.join(folder, "test.txt")
        with open(train_path, "w") as f_train:
            for l in variant_train_lines[variant]:
                f_train.write(l + "\n")
        with open(test_path, "w") as f_test:
            for l in variant_test_lines[variant]:
                f_test.write(l + "\n")
        print(f"UK{variant}: Wrote {len(variant_train_lines[variant])} train lines to {train_path} and {len(variant_test_lines[variant])} test lines to {test_path}")

def main():
    base_folder = "/media/data1/ningtong/wzh/projects/data/Image"
    create_label_files(base_folder)

if __name__ == "__main__":
    main()
