#!/usr/bin/env python3
"""
Comprehensive evaluation and visualization suite:
1. Convergence Plots (loss, accuracy, uncertainty separation across epochs)
2. ROC Curves for Out-of-Distribution (OOD) Detection using Epistemic Uncertainty
3. Sensor Data Visualizations comparing Nominal clean signals against Faulty ones
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from edl_model import FineTunedEvidentialClassifier1D
from fault_injection import (
    apply_gaussian_noise,
    apply_multiplicative_bias,
    apply_sensor_dropout,
)
from husky_dataloader import HuskyDataset, CHANNELS


def plot_convergence(
    history_path: str = "checkpoints/finetune_history.json",
    output_path: str = "plots/edl_convergence_plots.png",
):
    """Plots training and validation convergence curves."""
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Losses
    ax = axes[0]
    ax.plot(epochs, history["total_loss"], label="Total Loss", color="#1f77b4", linewidth=2.2)
    ax.plot(epochs, history["id_loss"], label="In-Distribution (ID) Loss", color="#2ca02c", linestyle="--", linewidth=1.8)
    ax.plot(epochs, history["ood_loss"], label="OOD Penalty Loss", color="#d62728", linestyle=":", linewidth=1.8)
    ax.set_title("Training Loss Convergence", fontweight="bold", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    # 2. Accuracies
    ax = axes[1]
    train_acc = [a * 100 for a in history["train_accuracy"]]
    val_acc = [a * 100 for a in history["val_clean_accuracy"]]
    ax.plot(epochs, train_acc, label="Train Accuracy (Clean)", color="#2ca02c", linewidth=2.2)
    ax.plot(epochs, val_acc, label="Held-Out Test Accuracy (Clean)", color="#1f77b4", marker="o", markersize=4, linewidth=2.0)
    ax.set_title("Classification Accuracy Convergence", fontweight="bold", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(40, 103)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    # 3. Uncertainty Divergence (ID vs OOD)
    ax = axes[2]
    clean_u = [u * 100 for u in history["val_clean_uncertainty"]]
    ood_u = [u * 100 for u in history["val_ood_uncertainty"]]
    ax.plot(epochs, clean_u, label="In-Distribution (Clean Test) Uncertainty", color="#2ca02c", linewidth=2.2)
    ax.plot(epochs, ood_u, label="Out-of-Distribution Uncertainty", color="#e7298a", linewidth=2.2)
    ax.axhline(90, color="crimson", linestyle="--", linewidth=1.5, label="Goal Target (90% Uncertainty)")
    ax.set_title("Epistemic Uncertainty Divergence", fontweight="bold", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Epistemic Uncertainty (%)", fontsize=11)
    ax.set_ylim(20, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Saved convergence plots to: {output_path}")


def plot_roc_curves(
    model: FineTunedEvidentialClassifier1D,
    test_loader: DataLoader,
    device: torch.device,
    output_path: str = "plots/edl_roc_curves.png",
):
    """Computes and plots ROC curves for OOD detection using predicted uncertainty."""
    # 1. Clean (In-Distribution) baseline uncertainty: y_true = 0
    clean_uncertainties = []
    with torch.no_grad():
        for b in test_loader:
            x = b["data"].to(device)
            out = model.predict_with_uncertainty(x)
            clean_uncertainties.extend(out["uncertainty"].cpu().numpy())
    clean_u = np.array(clean_uncertainties)
    n_clean = len(clean_u)

    # 2. OOD test sets: y_true = 1
    ood_sets = [
        ("Sensor Dropout (70%)", lambda x: apply_sensor_dropout(x, 0.7, mode="channel"), "#e7298a", "-"),
        ("Multiplicative Bias (100%)", lambda x: apply_multiplicative_bias(x, 1.0), "#d95f02", "-"),
        ("Gaussian Noise (120%)", lambda x: apply_gaussian_noise(x, 1.2), "#7570b3", "-"),
        ("Random Sensor Noise", lambda x: torch.randn_like(x) * 3.0, "#1b9e77", "-."),
        ("Mixed Noise + Dropout", lambda x: apply_sensor_dropout(apply_gaussian_noise(x, 1.0), 0.5, mode="channel"), "#e6ab02", ":"),
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    for name, corrupt_fn, color, ls in ood_sets:
        ood_u_list = []
        with torch.no_grad():
            for b in test_loader:
                x = corrupt_fn(b["data"].to(device))
                out = model.predict_with_uncertainty(x)
                ood_u_list.extend(out["uncertainty"].cpu().numpy())
        ood_u = np.array(ood_u_list)

        # Concatenate: clean = 0, OOD = 1
        y_true = np.concatenate([np.zeros(n_clean), np.ones(len(ood_u))])
        y_scores = np.concatenate([clean_u, ood_u])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr,
            label=f"{name} (AUROC = {roc_auc:.4f})",
            color=color,
            linestyle=ls,
            linewidth=2.4,
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Random Guess (AUROC = 0.50)")
    ax.set_title("ROC Curves: OOD Detection via Epistemic Uncertainty", fontweight="bold", fontsize=13)
    ax.set_xlabel("False Positive Rate (Clean Nominal falsely flagged as OOD)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Faulty/OOD correctly flagged)", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.03)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Saved ROC curves to: {output_path}")


def plot_sensor_visualizations(
    sample_tensor: torch.Tensor,
    terrain_name: str,
    output_path: str = "plots/husky_sensor_data_visualization.png",
):
    """
    Plots multi-channel sensor data comparing Nominal clean signals with:
    1. Multiplicative Bias (+100%)
    2. Gaussian Noise
    3. Sensor Dropout (30% / 70%)
    """
    x_clean = sample_tensor.clone().unsqueeze(0)  # (1, 10, 100)
    x_bias = apply_multiplicative_bias(x_clean, bias_pct=1.0)
    x_noise = apply_gaussian_noise(x_clean, noise_pct=0.8)
    x_dropout = apply_sensor_dropout(x_clean, dropout_pct=0.4, mode="channel")

    time_axis = np.linspace(0.0, 1.0, 100)  # 1.0s @ 100Hz

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    def plot_channels(ax, data_2d, title, is_corrupted=False):
        # Selected representative channels for clean presentation:
        # ax (accel X), az (accel Z), wz (yaw rate), curR (right motor current), velR (right wheel speed)
        channels_to_plot = [
            ("az (Acc Z)", 5, "#1f77b4"),
            ("ax (Acc X)", 3, "#ff7f0e"),
            ("wz (Yaw Rate)", 2, "#2ca02c"),
            ("curR (Motor Current)", 7, "#d62728"),
            ("velR (Wheel Speed)", 9, "#9467bd"),
        ]
        for ch_name, idx, col in channels_to_plot:
            ax.plot(time_axis, data_2d[idx, :], label=ch_name, color=col, linewidth=1.8, alpha=0.85)
        ax.set_title(title, fontweight="bold", fontsize=12, loc="left")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylabel("Amplitude / Sensor Value", fontsize=10)
        ax.legend(loc="upper right", ncol=5, frameon=True, fontsize=9)

    # 1. Clean Nominal
    plot_channels(axes[0], x_clean[0].numpy(), f"(A) Nominal Clean Signal (Terrain: {terrain_name.upper()}, Run 00)")

    # 2. Multiplicative Bias (+100%)
    plot_channels(axes[1], x_bias[0].numpy(), "(B) Fault: Multiplicative Bias (+100% Scale Inflation on all sensors)")

    # 3. Gaussian Noise
    plot_channels(axes[2], x_noise[0].numpy(), "(C) Fault: High-Frequency Gaussian Noise (Electromagnetic / Vibration Noise)")

    # 4. Sensor Dropout
    plot_channels(axes[3], x_dropout[0].numpy(), "(D) Fault: Sensor Channel Dropout (Dead channels / Disconnected Encoders zeroed to 0.0)")
    axes[3].set_xlabel("Time (seconds)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Saved sensor data visualizations to: {output_path}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  Generating Convergence Plots, ROC Curves, and Sensor Visualizations")
    print("=" * 80)

    # 1. Convergence Plots
    if Path("checkpoints/finetune_history.json").exists():
        plot_convergence()
    else:
        print("Note: checkpoints/finetune_history.json not found yet. Run fine_tune_edl.py first.")

    # 2. Load Model and Held-Out Test Set
    test_meta = torch.load("checkpoints/edl_test_data.pt", weights_only=False)
    dataset = HuskyDataset(channels_first=True)
    test_subset = Subset(dataset, test_meta["test_indices"])
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    ckpt = torch.load("checkpoints/edl_husky_finetuned.pth", map_location=device, weights_only=False)
    model = FineTunedEvidentialClassifier1D(num_classes=5).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. ROC Curves
    plot_roc_curves(model, test_loader, device)

    # 4. Sensor Data Visualizations
    sample = dataset[test_meta["test_indices"][0]]
    plot_sensor_visualizations(sample["data"], sample["terrain"])

    # 5. Print Final Verification Summary
    print("\n" + "=" * 80)
    print("  FINAL EVALUATION OF THE FINE-TUNED EDL MODEL ON TEST SETS")
    print("=" * 80)

    eval_conditions = [
        ("Nominal Clean Test", None),
        ("OOD: Sensor Dropout (70%)", lambda x: apply_sensor_dropout(x, 0.7, mode="channel")),
        ("OOD: Multiplicative Bias (100%)", lambda x: apply_multiplicative_bias(x, 1.0)),
        ("OOD: Severe Gaussian Noise (180%)", lambda x: apply_gaussian_noise(x, 1.8)),
        ("OOD: Random Ungrounded Noise", lambda x: torch.randn_like(x) * 3.0),
        ("OOD: Mixed Multi-Fault", lambda x: apply_sensor_dropout(apply_gaussian_noise(x, 1.0), 0.5, mode="channel")),
    ]

    print(f"  {'Dataset Condition':<34} | {'Accuracy':<10} | {'Mean Uncertainty':<18} | {'% Samples > 90% Uncert':<24} | {'Goal Status'}")
    print("  " + "-" * 105)

    for name, corrupt_fn in eval_conditions:
        u_list, corr, tot = [], 0, 0
        with torch.no_grad():
            for b in test_loader:
                x = b["data"].to(device)
                if corrupt_fn is not None:
                    x = corrupt_fn(x)
                out = model.predict_with_uncertainty(x)
                corr += (out["preds"] == b["label"].to(device)).sum().item()
                tot += len(b["label"])
                u_list.extend(out["uncertainty"].cpu().numpy())
        mean_u = float(np.mean(u_list))
        pct_above_90 = float(np.mean(np.array(u_list) > 0.90) * 100)
        acc = corr / tot
        status = "PASSED (Goal Met!)" if (mean_u >= 0.90 or name.startswith("Nominal")) else "INSPECT"
        print(f"  {name:<34} | {acc * 100:>8.2f}% | {mean_u * 100:>16.2f}% | {pct_above_90:>22.1f}% | {status}")

    print("=" * 80)


if __name__ == "__main__":
    main()
