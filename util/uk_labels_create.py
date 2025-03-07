import os
import shutil

def create_dataset_structure(base_folder):
    """
    Creates the image dataset folder structure:
      base_folder/dataset/data/1, 2, ... 7
    """
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
    """
    Reads list_patition_label.txt (each line: <filename> <label>),
    modifies the filename by inserting '_aligned' before '.jpg',
    and copies the file from base_folder/aligned to dataset/data/<label>.
    """
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
            
            # Process only labels 1 through 7
            if lbl not in [str(i) for i in range(1, 8)]:
                print(f"Skipping file {file_name}: label {lbl} is not in 1-7")
                continue
            
            # Modify file name: insert '_aligned' before '.jpg'
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
    """
    Creates the labels folder structure under:
      base_folder/dataset/labels/KU61/UK1, UK2, ... UK7
    For each UK folder, creates train.txt and test.txt.
    
    For each line in list_patition_label.txt (format: <filename> <label>):
      - The split is determined by checking if the filename contains "train" or "test" (case-insensitive).
      - For UKi:
          * In train.txt: include the line if split is "train" and the label is NOT equal to i.
          * In test.txt: include the line if split is "test". If the label equals i, change it to "8" before writing.
    """
    dataset_path = os.path.join(base_folder, "dataset")
    labels_folder = os.path.join(dataset_path, "labels")
    ku61_folder = os.path.join(labels_folder, "KU61")
    os.makedirs(ku61_folder, exist_ok=True)
    
    # Create UK1, UK2, ..., UK7 under KU61
    uk_folders = {}
    for i in range(1, 8):
        folder = os.path.join(ku61_folder, f"UK{i}")
        os.makedirs(folder, exist_ok=True)
        uk_folders[str(i)] = folder
    
    # Initialize dictionaries to store lines for each UK variant
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
            
            # Determine split from the file name (case-insensitive)
            lower_name = file_name.lower()
            if "train" in lower_name:
                split = "train"
            elif "test" in lower_name:
                split = "test"
            else:
                print(f"Skipping line because split cannot be determined: {orig_line}")
                continue
            
            # For each UK variant from 1 to 7
            for i in range(1, 8):
                variant = str(i)
                if split == "train":
                    # For train.txt: include if label is NOT equal to the variant
                    if lbl != variant:
                        variant_train_lines[variant].append(orig_line)
                elif split == "test":
                    # For test.txt: if label equals the variant, change it to "8"
                    if lbl == variant:
                        new_line = " ".join([file_name, "8"])
                        variant_test_lines[variant].append(new_line)
                    else:
                        variant_test_lines[variant].append(orig_line)
    
    # Write out the collected lines into train.txt and test.txt for each UK folder
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
    # Set the base folder path where the 'aligned' folder and 'list_patition_label.txt' reside.
    base_folder = "/media/data1/ningtong/wzh/projects/data/Image"
    
    # Step 1: Create dataset structure for images.
   ## data_path = create_dataset_structure(base_folder)
    
    # Step 2: Copy image files from 'aligned' based on list_patition_label.txt.
  ##  copy_files_according_to_label(base_folder, data_path)
    
    # Step 3: Create label files with the specified structure and modifications.
    create_label_files(base_folder)

if __name__ == "__main__":
    main()
