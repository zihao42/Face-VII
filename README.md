# Face-VII
USC CSCI 535 SPRING 2025 Group Project


# Set-up

python-version: 3.10

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

Run Training Command Example:

accelerate launch --num_processes 1 --mixed_precision bf16 --dynamo_backend=no fusion_train.py --loss_type scheduled --csv_file ./datasets/RAVDESS/csv/multimodel/multimodal-combination-1.csv --output_dir ./weights/scheduled

Run Evaluation Command Example:

python ./evaluation_noevi.py --weights_dir ./weights/model-ce --output_dir ./output/output3 --batch_size 16
