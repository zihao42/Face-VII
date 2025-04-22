# Face-VII
USC CSCI 535 SPRING 2025 Group Project


# Set-up
create a directory for saving weights

set up the directory for RAF-DB dataset

python-version: 3.10

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

Run Command:
CUDA_VISIBLE_DEVICES=1,2,3,5 accelerate launch --num_processes 4 --mixed_precision bf16 --dynamo_backend=no fusion_train.py --loss_type ce
