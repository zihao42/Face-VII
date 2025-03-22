import argparse
import sys
import numpy as np
import concurrent.futures
import torch
from data import get_dataloaders
from predict import predict_image  # 引用 predict.py 中的 predict_image
from enn_head import EvidentialClassificationHead
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 全局配置参数
IMG_DIR = "/media/data1/ningtong/wzh/projects/data/Image/aligned"
LABEL_FILE = "/media/data1/ningtong/wzh/projects/data/Image/list_patition_label.txt"
UK_MODE = "61"  # 更新后的 UK mode
model_name_pretrained = "microsoft/swin-tiny-patch4-window7-224"
num_labels = 6   # 模型输出标签数（训练时设定为6）
num_gpus = 4     # 可用 GPU 数量
uk_list = ["sur", "fea", "dis", "hap", "sad", "ang", "neu"]  # 7 个未知类别

def reorder_weight_files(weight_files):
    """
    根据预定义的顺序 ["sur", "fea", "dis", "hap", "sad", "ang", "neu"]
    重新排序权重文件列表。
    如果某个标签没有在文件名中找到，则打印错误信息并退出。
    """
    expected_order = ["sur", "fea", "dis", "hap", "sad", "ang", "neu"]
    weight_dict = {}
    for wf in weight_files:
        found = False
        for label in expected_order:
            if label in wf:
                # 若多个文件包含同一标签，可以根据需求做进一步处理，这里默认只出现一次
                weight_dict[label] = wf
                found = True
                break
        if not found:
            print(f"Warning: 文件 {wf} 中未找到任何预期的标签标识！", file=sys.stderr)
    ordered_weights = []
    for label in expected_order:
        if label in weight_dict:
            ordered_weights.append(weight_dict[label])
        else:
            print(f"Error: 权重文件中缺失标签 {label} 对应的文件。", file=sys.stderr)
            sys.exit(1)
    #print(ordered_weights)
    return ordered_weights

def evaluate_weight(weight_file, uk, threshold, gpu_id):
    """
    对某个权重文件（对应某个 UK）进行评估：
      1. 使用 get_dataloaders 获取测试 DataLoader（测试集限定为 10 张图片）。
      2. 在指定 GPU 上加载模型和该权重（只加载一次）。
      3. 遍历测试集每个样本，调用 predict_image（传入预加载 model）进行预测，
         并统计 True Positive、False Negative、False Positive、True Negative。
      4. 返回评估指标及子进程中的所有打印日志。
    """
    import io
    from transformers import SwinForImageClassification

    log_stream = io.StringIO()
    def log(*args, **kwargs):
        print(*args, **kwargs, file=log_stream, flush=True)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    log(f"[GPU:{gpu_id}] Using device: {device} for weight: {weight_file}")

    # 使用 get_dataloaders 获取测试 DataLoader（第三个返回值）
    _, _, dataloader_test = get_dataloaders(IMG_DIR, LABEL_FILE, UK_MODE, uk)
    log(f"[GPU:{gpu_id}] Loaded test dataloader for UK: {uk} with {len(dataloader_test.dataset)} samples")

    # 加载模型（只加载一次）
    model = SwinForImageClassification.from_pretrained(
        model_name_pretrained,
        ignore_mismatched_sizes=True,
        num_labels=num_labels
    )
    state_dict = torch.load(weight_file, map_location=device)
    
    # check if enn head is available
    enn_head = None
    if 'evi_head_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
        # default use_bn=True, will revise in later versions
        enn_head = EvidentialClassificationHead(model.config.hidden_size, num_labels, use_bn=True)
        enn_head.load_state_dict(state_dict['evi_head_state_dict'])
        enn_head.to(device)
        enn_head.eval()
        log(f"[GPU:{gpu_id}] ENN head loaded.")
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    log(f"[GPU:{gpu_id}] Model loaded and set to eval mode.")

    # 初始化计数器
    TP = 0  # 实际未知且预测为未知
    FN = 0  # 实际未知但预测为已知
    TN = 0  # 实际已知且预测为已知
    FP = 0  # 实际已知但预测为未知
    total = 0
    unknown_total = 0
    known_total = 0

    for images, labels in dataloader_test:
        for image, gt in zip(images, labels):
            total += 1
            if gt != 8:
                expected = gt.item() + 1
                known_total += 1
            else:
                expected = 8
                unknown_total += 1

            # 调用 predict_image，传入预加载 model 避免重复加载
            predicted_class, _ = predict_image(weight_file, image, threshold, model=model, enn_head=enn_head)
            if expected == 8:
                if predicted_class == 8:
                    TP += 1
                else:
                    FN += 1
            else:
                if predicted_class == 8:
                    FP += 1
                else:
                    TN += 1

            log(f"[GPU:{gpu_id}] Processed sample: expected={expected}, predicted={predicted_class}")

    accuracy = (TP + TN) / total * 100 if total > 0 else 0
    tp_percent = (TP / unknown_total * 100) if unknown_total > 0 else 0
    fn_percent = (FN / unknown_total * 100) if unknown_total > 0 else 0
    tn_percent = (TN / known_total * 100) if known_total > 0 else 0
    fp_percent = (FP / known_total * 100) if known_total > 0 else 0

    result = {
        "weight": weight_file,
        "uk": uk,
        "accuracy": accuracy,
        "TP": TP, "TP%": tp_percent,
        "FN": FN, "FN%": fn_percent,
        "TN": TN, "TN%": tn_percent,
        "FP": FP, "FP%": fp_percent,
        "total": total,
        "unknown_total": unknown_total,
        "known_total": known_total
    }
    return result, log_stream.getvalue()

def main():
    parser = argparse.ArgumentParser(
        description="Parallel evaluation of open-set emotion recognition model using multiple GPUs and get_dataloaders. "
                    "Output is saved to a txt file named after the model name."
    )
    # 模型名称（用于输出及日志文件命名）
    parser.add_argument("model_name", type=str, help="Name of the model (used for output file name)")
    # 期望传入 7 个权重文件（对应 7 个 UK）
    parser.add_argument("weights", nargs=7, help="7 weight files (.pth) to evaluate")
    args = parser.parse_args()

    # 对传入的权重文件进行排序，确保顺序为：sur, fea, dis, hap, sad, ang, neu
    weight_files = reorder_weight_files(args.weights)

    model_name_print = args.model_name

    # 创建一个以模型名称命名的日志文件
    output_filename = f"{model_name_print}.txt"
    log_file = open(output_filename, "w", encoding="utf-8")

    # 定义 Tee 类，同时输出到终端和日志文件
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    sys.stdout = Tee(sys.stdout, log_file)

    print("=" * 80)
    print(f"Model Name: {model_name_print}")
    print("=" * 80)

    # 对多个阈值进行评估
    thresholds = np.arange(0.05, 1.0, 0.1)
    for thresh in thresholds:
        print("=" * 80)
        print(f"Threshold: {thresh:.2f}")
        all_metrics = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, weight_file in enumerate(weight_files):
                uk = uk_list[i]
                gpu_id = i % num_gpus  # 简单分配 GPU
                futures.append(executor.submit(evaluate_weight, weight_file, uk, thresh, gpu_id))
            for future in concurrent.futures.as_completed(futures):
                try:
                    result, logs = future.result(timeout=300)
                    print(logs, flush=True)
                    all_metrics.append(result)
                    print(f"Weight: {result['weight']} (UK: {result['uk']}) | Accuracy: {result['accuracy']:.2f}% | "
                          f"TP: {result['TP']} ({result['TP%']:.2f}%) | FN: {result['FN']} ({result['FN%']:.2f}%) | "
                          f"TN: {result['TN']} ({result['TN%']:.2f}%) | FP: {result['FP']} ({result['FP%']:.2f}%)",
                          flush=True)
                except Exception as e:
                    print("Error in one of the tasks:", e, flush=True)
        mean_accuracy = np.mean([m['accuracy'] for m in all_metrics])
        total_tp = np.mean([m['TP'] for m in all_metrics])
        total_fn = np.mean([m['FN'] for m in all_metrics])
        total_tn = np.mean([m['TN'] for m in all_metrics])
        total_fp = np.mean([m['FP'] for m in all_metrics])
        print("-" * 80)
        print(f"Metrics for Threshold {thresh:.2f}: Accuracy: {mean_accuracy:.2f}% | "
                f"TP: {total_tp:.2f} | FN: {total_fn:.2f} | "
                f"TN: {total_tn:.2f} | FP: {total_fp:.2f}", flush=True)
        print("-" * 80)

    log_file.close()

if __name__ == "__main__":
    main()
