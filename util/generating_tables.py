import re
import argparse
import pandas as pd

def parse_txt(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    data = {}
    current_threshold = None

    for line in lines:
        match = re.search(r"Threshold: ([0-9.]+)", line)
        if match:
            current_threshold = float(match.group(1))
            if current_threshold not in data:
                data[current_threshold] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "Accuracy": None}

        match = re.search(r"TP: (\d+) .*? FN: (\d+) .*? TN: (\d+) .*? FP: (\d+)", line)
        if match and current_threshold is not None:
            tp, fn, tn, fp = map(int, match.groups())
            data[current_threshold]["TP"] += tp
            data[current_threshold]["FN"] += fn
            data[current_threshold]["TN"] += tn
            data[current_threshold]["FP"] += fp

        match = re.search(r"Mean Metrics for Threshold [0-9.]+: Accuracy: ([0-9.]+)%", line)
        if match and current_threshold is not None:
            data[current_threshold]["Accuracy"] = float(match.group(1)) / 100

    return data


def compute_metrics(data):
    metrics = {}
    for thr, counts in data.items():
        tp = counts["TP"]
        tn = counts["TN"]
        fp = counts["FP"]
        fn = counts["FN"]
        if counts["Accuracy"] is not None:
            accuracy = counts["Accuracy"]
        else:
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics[thr] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall}
    return metrics


def frange(start, stop, step):
    values = []
    while start <= stop + 1e-9:
        values.append(round(start, 2))
        start += step
    return values


def generate_table(metrics, thresholds):
    accuracy_list = []
    precision_list = []
    recall_list = []
    col_labels = []
    
    for thr in thresholds:
        col_labels.append(str(thr))
        if thr in metrics:
            m = metrics[thr]
            accuracy_list.append(round(m["Accuracy"] * 100, 2))
            precision_list.append(round(m["Precision"] * 100, 2))
            recall_list.append(round(m["Recall"] * 100, 2))
        else:
            accuracy_list.append(None)
            precision_list.append(None)
            recall_list.append(None)

    df = pd.DataFrame(
        {
            "Accuracy": accuracy_list,
            "Precision": precision_list,
            "Recall": recall_list
        },
        index=col_labels
    ).transpose()
    
    return df

def main(file_paths):
    thresholds_range = frange(0.05, 0.95, 0.1)
    
    for file_path in file_paths:
        print(f"\nProcessing file: {file_path}")
        parsed_data = parse_txt(file_path)
        metrics = compute_metrics(parsed_data)
        table = generate_table(metrics, thresholds_range)
        print("Metrics Table (values in %):")
        print(table)
        print("-" * 50)

        latex_table = table.to_latex(index=True, float_format="%.2f")
        print("LaTeX Table Code (for Overleaf):")
        print(latex_table)
        print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a table with Accuracy, Precision, and Recall for each threshold (0.05 to 0.95, step 0.1) from a txt file, and output LaTeX code."
    )
    parser.add_argument("files", nargs="+", help="Path(s) to the txt file(s)")
    args = parser.parse_args()
    main(args.files)
