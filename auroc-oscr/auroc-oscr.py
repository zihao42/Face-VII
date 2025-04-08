#!/usr/bin/env python
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os

def parse_txt(file_path):
    """
    Parse the txt file and extract, for each threshold, all weight file's TP, TN, FP, FN counts (accumulated),
    and extract the Mean Accuracy.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    data = {}
    current_threshold = None

    for line in lines:
        # Extract Threshold
        match = re.search(r"Threshold: ([0-9.]+)", line)
        if match:
            current_threshold = float(match.group(1))
            if current_threshold not in data:
                data[current_threshold] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "Accuracy": None}

        # Extract TP, TN, FP, FN (accumulate for each weight file)
        match = re.search(r"TP: (\d+) .*? FN: (\d+) .*? TN: (\d+) .*? FP: (\d+)", line)
        if match and current_threshold is not None:
            tp, fn, tn, fp = map(int, match.groups())
            data[current_threshold]["TP"] += tp
            data[current_threshold]["FN"] += fn
            data[current_threshold]["TN"] += tn
            data[current_threshold]["FP"] += fp

        # Extract Mean Accuracy
        match = re.search(r"Mean Metrics for Threshold [0-9.]+: Accuracy: ([0-9.]+)%", line)
        if match and current_threshold is not None:
            data[current_threshold]["Accuracy"] = float(match.group(1)) / 100  # convert to decimal

    return data

def compute_metrics(data):
    """
    Compute TPR, FPR, Accuracy, and rejection rate for ROC and OSCR curves.
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

        print(f"{threshold:<10.2f}{accuracy:<10.4f}{tp:<10}{tn:<10}{fp:<10}{fn:<10}")

        thresholds.append(threshold)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        accuracy_list.append(accuracy)
        rejection_list.append(rejection)

    print("-" * 60)
    return (np.array(thresholds), np.array(jjtpr_list), np.array(fpr_list),
            np.array(accuracy_list), np.array(rejection_list))

def calculate_bestfit_auc(x_vals, y_vals, degree=3, force_endpoints=False):
    """
    If force_endpoints is True, add endpoints (0,0) and (1,1) to x_vals and y_vals.
    Then fit a polynomial of the specified degree and compute the area under the curve by integrating it.
    """
    if force_endpoints:
        x_vals = np.concatenate(([0], x_vals, [1]))
        y_vals = np.concatenate(([0], y_vals, [1]))
    # Fit polynomial
    poly_coeff = np.polyfit(x_vals, y_vals, degree)
    poly_fit = np.poly1d(poly_coeff)
    # Integrate polynomial between min and max of x_vals (should be 0 and 1 if forced)
    area = poly_fit.integ()(np.max(x_vals)) - poly_fit.integ()(np.min(x_vals))
    return area, poly_fit

def plot_combined_curves(models, x_arrays, y_arrays, output_path, title, xlabel, ylabel, labels, force_endpoints=False):
    """
    Plot curves for multiple models and save as JPEG.
    For the AUROC curve, if force_endpoints is True, force endpoints (0,0) and (1,1) and fit a best-fit curve.
    """
    plt.figure(figsize=(6, 5))
    auc_values = {}
    for i, model in enumerate(models):
        x_vals = x_arrays[i]
        y_vals = y_arrays[i]
        if force_endpoints:
            x_vals = np.concatenate(([0], x_vals, [1]))
            y_vals = np.concatenate(([0], y_vals, [1]))
        # Fit best-fit polynomial if there are enough points
        if len(x_vals) >= 4:
            area, poly_fit = calculate_bestfit_auc(x_vals, y_vals, degree=3, force_endpoints=False)
            auc_values[labels[i]] = area
            # Generate smooth line for best-fit
            x_new = np.linspace(np.min(x_vals), np.max(x_vals), 100)
            y_new = poly_fit(x_new)
            plt.plot(x_new, y_new, linestyle='-', label=f"{labels[i]} (fit, AUC={area:.3f})")
            # Plot original points for reference
            plt.plot(x_vals, y_vals, marker='o', linestyle='None', color=plt.gca().lines[-1].get_color())
        else:
            plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=labels[i])
    # For AUROC, plot diagonal line
    if title.lower().startswith("auroc"):
        plt.plot([0, 1], [0, 1], linestyle='--', color="gray")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, format='jpeg', dpi=300)
    plt.close()
    print(f"{title} saved as {output_path}")
    return auc_values

if __name__ == "__main__":
    # The following file names are assumed. Adjust if needed.
    file_paths = ["baseline_2.txt", "variance_2.txt", "evidential_2.txt"]
    model_names = ["Base Model", "Variance Model", "Evidential Model", ]
    
    all_tprs, all_fprs, all_accuracies, all_rejections = [], [], [], []
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        data = parse_txt(file_path)
        _, tpr_list, fpr_list, accuracy_list, rejection_list = compute_metrics(data)
        all_tprs.append(tpr_list)
        all_fprs.append(fpr_list)
        all_accuracies.append(accuracy_list)
        all_rejections.append(rejection_list)
    
    # Plot AUROC (FPR vs. TPR) with best-fit curves.
    # Force endpoints for the AUROC curve.
    auroc_auc_values = plot_combined_curves(
        model_names, all_fprs, all_tprs,
        "auroc.jpeg", "AUROC Curve",
        "False Positive Rate (FPR)", "True Positive Rate (TPR)", model_names,
        force_endpoints=True
    )
    
    # Plot OSCR (Rejection vs. Accuracy) with best-fit curves.
    oscr_auc_values = plot_combined_curves(
        model_names, all_rejections, all_accuracies,
        "oscr.jpeg", "OSCR Curve",
        "Rejection Rate (1 - FPR)", "Classification Accuracy", model_names,
        force_endpoints=False  # Usually, OSCR does not need forced endpoints.
    )
    
    print("\nCalculated AUROC areas from best-fit curves:")
    for model, auc_val in auroc_auc_values.items():
        print(f"{model}: AUROC = {auc_val:.3f}")
