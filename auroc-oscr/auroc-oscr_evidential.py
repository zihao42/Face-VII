#!/usr/bin/env python
import argparse
import numpy as np
import torch
from torchmetrics import AUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def compute_auroc(scores_path, labels_path):
    scores = np.load(scores_path)
    labels = np.load(labels_path)
    scores_tensor = torch.tensor(scores, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.int)
    auroc_metric = AUROC(task="binary")
    auroc_value = auroc_metric(scores_tensor, labels_tensor)
    return auroc_value.item()

def compute_oscr_curve(scores, labels, known_acc_thresholds, num_thresholds=100):
    """
    Compute OSCR curve given raw unknown scores and binary labels (for unknown detection),
    along with known classification accuracy at different thresholds (provided by known_acc_thresholds).
    Here, we assume that the rejection rate for unknown samples can be derived from the ROC.
    This is a placeholder implementation; adjust the computation based on your OSCR definition.
    """
    # Get thresholds from min to max of scores
    thresholds = np.linspace(np.min(scores), np.max(scores), num_thresholds)
    rejection_rates = []
    accuracies = []
    for thresh in thresholds:
        # For unknown detection:
        # Predict unknown if score > thresh (adjust inequality if needed)
        pred_unknown = (scores > thresh).astype(int)
        # Rejection rate can be defined as proportion of unknown samples correctly rejected,
        # for example, 1 - false positive rate for unknown detection.
        # Compute FPR for unknown samples:
        FP = np.sum((pred_unknown == 1) & (labels == 0))
        TN = np.sum((pred_unknown == 0) & (labels == 0))
        rejection_rate = 1 - (FP / (FP + TN)) if (FP+TN) > 0 else 0
        rejection_rates.append(rejection_rate)
        
        # For known classification accuracy,
        # we assume known_acc_thresholds is a function or an array that gives known accuracy at this threshold.
        # For simplicity, here we interpolate from known_acc_thresholds if provided as a tuple (thresh_array, acc_array).
        # Otherwise, we can simply set it to a constant (placeholder).
        accuracies.append(np.interp(thresh, known_acc_thresholds[0], known_acc_thresholds[1]))
        
    return thresholds, np.array(rejection_rates), np.array(accuracies)

def plot_curves(thresholds, rejection_rates, accuracies, auroc_value, output_path):
    plt.figure(figsize=(6,5))
    plt.plot(rejection_rates, accuracies, marker='o', linestyle='-', label=f"OSCR Curve")
    plt.xlabel("Rejection Rate (1 - FPR)")
    plt.ylabel("Classification Accuracy (Known)")
    plt.title(f"OSCR Curve (AUROC: {auroc_value:.3f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, format="jpeg", dpi=300)
    plt.close()
    print(f"OSCR curve saved as {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute AUROC and OSCR curves from evidential raw unknown scores and binary labels."
    )
    parser.add_argument("--scores", type=str, required=True,
                        help="Path to a numpy file (.npy) containing raw unknown scores (1D array)")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to a numpy file (.npy) containing binary ground truth labels (1 for unknown, 0 for known)")
    # For OSCR, we also need known accuracy info across thresholds.
    # Here we assume a numpy file with two columns: first column = thresholds, second = known accuracy.
    parser.add_argument("--known_acc", type=str, required=False,
                        help="Path to a numpy file (.npy) containing thresholds and known accuracy for OSCR computation")
    parser.add_argument("--oscr_out", type=str, default="oscr.jpeg",
                        help="Path to save the OSCR curve plot (JPEG)")
    args = parser.parse_args()

    # Compute AUROC using raw scores and binary labels
    auroc_value = compute_auroc(args.scores, args.labels)
    print(f"Computed AUROC: {auroc_value:.3f}")

    # Load the raw scores and binary labels
    scores = np.load(args.scores)
    labels = np.load(args.labels)

    # For OSCR, if known classification accuracy is provided, load it.
    if args.known_acc:
        known_data = np.load(args.known_acc, allow_pickle=True)
        # Assume known_data is a tuple: (thresholds, accuracies)
        known_acc_thresholds = known_data
    else:
        # Otherwise, use a placeholder: assume constant known accuracy of 90%
        thresholds_placeholder = np.linspace(np.min(scores), np.max(scores), 100)
        accuracies_placeholder = np.full_like(thresholds_placeholder, 0.90)
        known_acc_thresholds = (thresholds_placeholder, accuracies_placeholder)

    # Compute OSCR curve: we derive rejection rate and known accuracy at each threshold.
    thresholds, rejection_rates, accuracies = compute_oscr_curve(scores, labels, known_acc_thresholds)
    
    # Plot OSCR curve
    plot_curves(thresholds, rejection_rates, accuracies, auroc_value, args.oscr_out)

if __name__ == "__main__":
    main()
