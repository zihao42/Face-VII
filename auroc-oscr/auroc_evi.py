import numpy as np
import torch
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def main():
    unknown_scores = np.load(f"unknown_scores.npy")
    unknown_labels = np.load(f"unknown_labels.npy")

    scores_tensor = torch.tensor(unknown_scores)
    labels_tensor = torch.tensor(unknown_labels).long()

    auroc_metric = torchmetrics.AUROC(task="binary")
    auroc_score = auroc_metric(scores_tensor, labels_tensor)
    print("AUROC Score:", auroc_score.item())

    fpr, tpr, roc_thresholds = roc_curve(unknown_labels, unknown_scores, pos_label=1)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auroc_score.item():.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("auroc_evi.jpeg")
    plt.close()
    

if __name__ == '__main__':
    main()
