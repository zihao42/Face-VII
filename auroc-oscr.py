import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os

def parse_txt(file_path):
    """
    解析 txt 文件，提取每个 threshold 下所有 weight file 的 TP, TN, FP, FN，并计算总数。
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    data = {}
    current_threshold = None

    for line in lines:
        # 提取 Threshold
        match = re.search(r"Threshold: ([0-9.]+)", line)
        if match:
            current_threshold = float(match.group(1))
            if current_threshold not in data:
                data[current_threshold] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "Accuracy": None}

        # 提取 TP, TN, FP, FN（从每个 weight filepath 统计后累加）
        match = re.search(r"TP: (\d+) .*? FN: (\d+) .*? TN: (\d+) .*? FP: (\d+)", line)
        if match and current_threshold is not None:
            tp, fn, tn, fp = map(int, match.groups())
            data[current_threshold]["TP"] += tp
            data[current_threshold]["FN"] += fn
            data[current_threshold]["TN"] += tn
            data[current_threshold]["FP"] += fp

        # 提取 Mean Accuracy
        match = re.search(r"Mean Metrics for Threshold [0-9.]+: Accuracy: ([0-9.]+)%", line)
        if match and current_threshold is not None:
            data[current_threshold]["Accuracy"] = float(match.group(1)) / 100  # 转换为小数

    return data

def compute_metrics(data):
    """
    计算 AUROC 和 OSCR 需要的 TPR, FPR, Accuracy-Rejection 曲线。
    """
    thresholds = []
    tpr_list = []
    fpr_list = []
    accuracy_list = []
    rejection_list = []

    print("\n======= Mean Metrics Per Threshold =======")
    print(f"{'Threshold':<10}{'Accuracy':<10}{'TP':<10}{'TN':<10}{'FP':<10}{'FN':<10}")
    print("-" * 60)

    for threshold, values in sorted(data.items()):
        tp, fn, tn, fp = values["TP"], values["FN"], values["TN"], values["FP"]
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        accuracy = values["Accuracy"]
        rejection = 1 - fpr  # Open-set rejection rate

        # 打印 Mean Metrics
        print(f"{threshold:<10.2f}{accuracy:<10.4f}{tp:<10}{tn:<10}{fp:<10}{fn:<10}")

        thresholds.append(threshold)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        accuracy_list.append(accuracy)
        rejection_list.append(rejection)

    print("-" * 60)
    return np.array(thresholds), np.array(tpr_list), np.array(fpr_list), np.array(accuracy_list), np.array(rejection_list)

def plot_auroc(fpr, tpr, output_path):
    """
    绘制 AUROC 曲线，并保存为 JPEG。
    """
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, marker='o', linestyle='-', label=f'AUROC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color="gray")  # 对角线
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("AUROC Curve")
    plt.legend()
    plt.grid(True)

    # 保存 AUROC 曲线
    plt.savefig(output_path, format='jpeg', dpi=300)
    plt.close()

    print(f"\nAUROC Score: {roc_auc:.3f}")
    print(f"AUROC curve saved as {output_path}")

def plot_oscr(rejection, accuracy, output_path):
    """
    绘制 OSCR 曲线，并保存为 JPEG。
    """
    oscr_score = auc(rejection, accuracy)
    plt.figure(figsize=(6, 5))
    plt.plot(rejection, accuracy, marker='s', linestyle='-', label=f'OSCR = {oscr_score:.3f}')
    plt.xlabel("Rejection Rate (1 - FPR)")
    plt.ylabel("Classification Accuracy")
    plt.title("OSCR Curve")
    plt.legend()
    plt.grid(True)

    # 保存 OSCR 曲线
    plt.savefig(output_path, format='jpeg', dpi=300)
    plt.close()

    print(f"\nOSCR Score: {oscr_score:.3f}")
    print(f"OSCR curve saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_auroc_oscr.py <input_file.txt>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # 获取文件名（去掉路径和扩展名）

    print(f"Processing file: {file_path}")

    # 解析文件
    data = parse_txt(file_path)

    # 计算 TPR, FPR, Accuracy-Rejection 曲线，并打印 Mean Metrics
    thresholds, tpr_list, fpr_list, accuracy_list, rejection_list = compute_metrics(data)

    # 生成文件名
    auroc_output = f"{file_name}-auroc.jpeg"
    oscr_output = f"{file_name}-oscr.jpeg"

    # 绘制 AUROC 和 OSCR，并保存图像
    plot_auroc(fpr_list, tpr_list, auroc_output)
    plot_oscr(rejection_list, accuracy_list, oscr_output)
