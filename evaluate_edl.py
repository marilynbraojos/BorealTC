#!/usr/bin/env python3
"""
Evaluation script for Evidential Deep Learning (EDL) model on Husky A200 terrain data.
Computes classification metrics, uncertainty calibration, and rejection curve analysis.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

from husky_dataloader import HuskyDataset
from edl_model import EvidentialClassifier1D


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Computes Multi-class Brier Score (lower is better, range [0, 2])."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def evaluate(
    checkpoint_path: str = "checkpoints/edl_husky.pth",
    test_meta_path: str = "checkpoints/edl_test_data.pt",
    output_plot: str = "edl_evaluation_results.png",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("  Evidential Deep Learning Evaluation & Uncertainty Quantification")
    print("=" * 75)

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = checkpoint["num_classes"]
    in_channels = checkpoint["in_channels"]
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    model = EvidentialClassifier1D(in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded EDL model from: {checkpoint_path}")

    # 2. Load Test Partition
    test_meta = torch.load(test_meta_path, weights_only=False)
    data_dir = test_meta.get("data_dir", "data/borealtc")
    test_indices = test_meta["test_indices"]

    dataset = HuskyDataset(
        data_dir=data_dir,
        window_size=100,
        step_size=50,
        channels_first=True,
    )
    test_subset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    print(f"Loaded {len(test_indices)} test samples across classes: {class_names}")

    # 3. Model Inference with Uncertainty
    all_targets = []
    all_preds = []
    all_probs = []
    all_uncertainties = []
    all_confidences = []
    all_evidences = []
    all_terrains = []
    all_run_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["data"].to(device)
            targets = batch["label"].cpu().numpy()
            terrains = batch["terrain"]
            run_ids = batch["run_id"]

            out = model.predict_with_uncertainty(inputs)
            
            all_targets.extend(targets)
            all_preds.extend(out["preds"].cpu().numpy())
            all_probs.extend(out["probs"].cpu().numpy())
            all_uncertainties.extend(out["uncertainty"].cpu().numpy())
            all_confidences.extend(out["confidence"].cpu().numpy())
            all_evidences.extend(out["evidence"].cpu().numpy())
            all_terrains.extend(terrains)
            all_run_ids.extend(run_ids)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs = np.array(all_probs)
    uncertainties = np.array(all_uncertainties)
    confidences = np.array(all_confidences)
    is_correct = (y_pred == y_true)

    # 4. Standard Classification Metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "-" * 75)
    print(" [1] CLASSIFICATION PERFORMANCE METRICS")
    print("-" * 75)
    print(f"  • Overall Accuracy   : {accuracy * 100:.2f}%")
    print(f"  • Macro F1-Score     : {macro_f1:.4f}")
    print(f"  • Weighted F1-Score  : {weighted_f1:.4f}")

    print("\n Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(" Confusion Matrix (Rows: True, Cols: Pred):")
    header = "  " + "".join([f"{name[:7]:>8}" for name in class_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join([f"{val:>8}" for val in row])
        print(f"  {class_names[i][:6]:<6} {row_str}")

    # 5. Uncertainty Quantification (UQ) Metrics
    ece = compute_ece(probs, y_true)
    brier = compute_brier_score(probs, y_true, num_classes)

    mean_uncert_correct = np.mean(uncertainties[is_correct]) if np.any(is_correct) else 0.0
    mean_uncert_wrong = np.mean(uncertainties[~is_correct]) if np.any(~is_correct) else 0.0
    uncert_ratio = mean_uncert_wrong / (mean_uncert_correct + 1e-8)

    print("\n" + "-" * 75)
    print(" [2] EVIDENTIAL UNCERTAINTY QUANTIFICATION (UQ) METRICS")
    print("-" * 75)
    print(f"  • Mean Epistemic Uncertainty (Overall): {np.mean(uncertainties):.4f}  (Range: [0=certain, 1=vacuous])")
    print(f"  • Mean Uncertainty on CORRECT preds   : {mean_uncert_correct:.4f}")
    print(f"  • Mean Uncertainty on WRONG preds     : {mean_uncert_wrong:.4f}")
    print(f"  • Uncertainty Ratio (Wrong / Correct) : {uncert_ratio:.2f}x")
    print(f"    (Higher ratio means model is appropriately uncertain when it makes mistakes)")
    print(f"  • Expected Calibration Error (ECE)    : {ece:.4f}  (Lower is better, [0, 1])")
    print(f"  • Multi-Class Brier Score             : {brier:.4f}  (Lower is better, [0, 2])")

    # 6. Uncertainty Rejection Analysis (Selective Classification)
    print("\n" + "-" * 75)
    print(" [3] UNCERTAINTY REJECTION / FILTERING ANALYSIS")
    print(" (Simulating safety filter: withholding predictions with highest uncertainty)")
    print("-" * 75)
    print(f"  {'Retained %':<12} | {'Remaining Samples':<18} | {'Accuracy':<10} | {'Max Uncertainty':<15}")
    print("  " + "-" * 62)

    retention_fractions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    rejection_accuracies = []
    
    # Sort samples by ascending uncertainty (most confident first)
    sort_order = np.argsort(uncertainties)
    sorted_correct = is_correct[sort_order]
    sorted_uncert = uncertainties[sort_order]

    for frac in retention_fractions:
        k = int(np.ceil(len(y_true) * frac))
        retained_acc = np.mean(sorted_correct[:k])
        max_u = sorted_uncert[k - 1]
        rejection_accuracies.append(retained_acc)
        print(f"  {int(frac * 100):>3}%         | {k:<18} | {retained_acc * 100:>6.2f}%   | {max_u:<15.4f}")

    # 7. Sample Predictions Inspection Table
    print("\n" + "-" * 75)
    print(" [4] SAMPLE PREDICTIONS & UNCERTAINTY ESTIMATES")
    print("-" * 75)
    sample_header = f"  {'Sample':<6} | {'True Terrain':<12} | {'Pred Terrain':<12} | {'Confidence':<10} | {'Uncertainty':<12} | {'Status'}"
    print(sample_header)
    print("  " + "-" * (len(sample_header) - 2))

    for idx in range(min(8, len(y_true))):
        true_name = class_names[y_true[idx]]
        pred_name = class_names[y_pred[idx]]
        conf = confidences[idx]
        unc = uncertainties[idx]
        status = "CORRECT" if is_correct[idx] else "INCORRECT"
        print(f"  {idx:<6} | {true_name:<12} | {pred_name:<12} | {conf:10.3f} | {unc:12.4f} | {status}")

    # 8. Generate Summary Plots
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # (A) Confusion Matrix Heatmap
        ax = axes[0, 0]
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center", size="large")
        fig.colorbar(cax, ax=ax)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=30, ha="left")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix", pad=20, fontweight="bold")

        # (B) Uncertainty Distribution: Correct vs Incorrect
        ax = axes[0, 1]
        if np.any(is_correct):
            ax.hist(uncertainties[is_correct], bins=15, alpha=0.6, label=f"Correct (mean={mean_uncert_correct:.3f})", color="forestgreen", density=True)
        if np.any(~is_correct):
            ax.hist(uncertainties[~is_correct], bins=15, alpha=0.6, label=f"Wrong (mean={mean_uncert_wrong:.3f})", color="crimson", density=True)
        ax.set_xlabel("Epistemic Uncertainty u in [0, 1]")
        ax.set_ylabel("Density")
        ax.set_title("Uncertainty: Correct vs Incorrect", fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        # (C) Uncertainty Rejection Curve
        ax = axes[1, 0]
        retained_pcts = [f * 100 for f in retention_fractions]
        ax.plot(retained_pcts, [a * 100 for a in rejection_accuracies], marker="o", color="royalblue", linewidth=2)
        ax.set_xlabel("Retained Samples (%) [sorted by low uncertainty]")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Uncertainty-Based Rejection Curve", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        # (D) Confidence vs Accuracy Calibration (Reliability Diagram)
        ax = axes[1, 1]
        bins = np.linspace(0, 1, 11)
        bin_accs = []
        bin_confs = []
        for i in range(10):
            in_b = (confidences > bins[i]) & (confidences <= bins[i+1])
            if np.any(in_b):
                bin_accs.append(np.mean(is_correct[in_b]))
                bin_confs.append(np.mean(confidences[in_b]))
            else:
                bin_accs.append(0.0)
                bin_confs.append((bins[i] + bins[i+1]) / 2)
        ax.bar(bins[:-1] + 0.05, bin_accs, width=0.08, alpha=0.6, color="teal", label="Empirical Accuracy", edgecolor="black")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})", fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_plot, dpi=150)
        print(f"\n[5] Saved evaluation figures to: {output_plot}")
    except Exception as e:
        print(f"\nCould not generate matplotlib figure: {e}")

    print("\n" + "=" * 75)
    print("  Evaluation finished successfully.")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EDL model on Husky A200 test set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/edl_husky.pth")
    parser.add_argument("--test-meta", type=str, default="checkpoints/edl_test_data.pt")
    parser.add_argument("--output-plot", type=str, default="edl_evaluation_results.png")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        test_meta_path=args.test_meta,
        output_plot=args.output_plot,
    )
