#!/usr/bin/env python3
"""
Evaluates the robustness and uncertainty quantification of the Evidential Deep Learning (EDL)
model across multiple faulty test sets:
1. Multiplicative Bias (0% to 100% in 10% increments)
2. Gaussian Noise (0% to 100% in 10% increments)
3. Sensor Dropout (0% to 100% in 10% increments)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from edl_model import EvidentialClassifier1D
from evaluate_edl import compute_brier_score, compute_ece
from fault_injection import (
    apply_gaussian_noise,
    apply_multiplicative_bias,
    apply_sensor_dropout,
)
from husky_dataloader import HuskyDataset


def evaluate_fault_sweep(
    checkpoint_path: str = "checkpoints/edl_husky.pth",
    test_meta_path: str = "checkpoints/edl_test_data.pt",
    output_plot: str = "fault_robustness_analysis.png",
    output_json: str = "fault_robustness_results.json",
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    print("=" * 80)
    print("  Evidential Deep Learning Fault Robustness & Uncertainty Analysis")
    print("=" * 80)

    # 1. Load Checkpoint and Model
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

    # 2. Load Held-Out Test Set
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
    print(f"Loaded held-out test set: {len(test_indices)} samples across classes: {class_names}")

    # Baseline evaluation (0% fault)
    fault_levels = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    fault_types = ["multiplicative_bias", "gaussian_noise", "sensor_dropout"]

    results = {
        ft: {
            "levels": [float(lvl) for lvl in fault_levels],
            "accuracy": [],
            "macro_f1": [],
            "mean_uncertainty": [],
            "mean_confidence": [],
            "ece": [],
            "brier": [],
        }
        for ft in fault_types
    }

    # 3. Run Sweeps for each fault type
    for fault_name in fault_types:
        print("\n" + "=" * 80)
        print(f"  FAULT MODE: {fault_name.replace('_', ' ').upper()}")
        print("=" * 80)
        print(f"{'Severity':<10} | {'Accuracy':<10} | {'Macro F1':<10} | {'Uncertainty':<14} | {'Confidence':<12} | {'ECE':<8}")
        print("-" * 72)

        for lvl in fault_levels:
            all_targets = []
            all_preds = []
            all_probs = []
            all_uncert = []
            all_conf = []

            for batch in test_loader:
                inputs = batch["data"].to(device)
                targets = batch["label"].cpu().numpy()

                # Inject fault
                if fault_name == "multiplicative_bias":
                    faulty_inputs = apply_multiplicative_bias(inputs, bias_pct=lvl)
                elif fault_name == "gaussian_noise":
                    faulty_inputs = apply_gaussian_noise(inputs, noise_pct=lvl, generator=gen)
                elif fault_name == "sensor_dropout":
                    faulty_inputs = apply_sensor_dropout(inputs, dropout_pct=lvl, mode="channel", generator=gen)
                else:
                    raise ValueError(f"Unknown fault: {fault_name}")

                with torch.no_grad():
                    out = model.predict_with_uncertainty(faulty_inputs)

                all_targets.extend(targets)
                all_preds.extend(out["preds"].cpu().numpy())
                all_probs.extend(out["probs"].cpu().numpy())
                all_uncert.extend(out["uncertainty"].cpu().numpy())
                all_conf.extend(out["confidence"].cpu().numpy())

            y_true = np.array(all_targets)
            y_pred = np.array(all_preds)
            probs = np.array(all_probs)
            uncert = np.array(all_uncert)
            conf = np.array(all_conf)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            mean_u = float(np.mean(uncert))
            mean_c = float(np.mean(conf))
            ece_val = compute_ece(probs, y_true)
            brier_val = compute_brier_score(probs, y_true, num_classes)

            results[fault_name]["accuracy"].append(float(acc))
            results[fault_name]["macro_f1"].append(float(f1))
            results[fault_name]["mean_uncertainty"].append(mean_u)
            results[fault_name]["mean_confidence"].append(mean_c)
            results[fault_name]["ece"].append(float(ece_val))
            results[fault_name]["brier"].append(float(brier_val))

            pct_str = f"{int(round(lvl * 100))}%"
            print(
                f"{pct_str:<10} | {acc * 100:>8.2f}% | {f1:>10.4f} | "
                f"{mean_u:>14.4f} | {mean_c:>12.4f} | {ece_val:>8.4f}"
            )

    # 4. Save JSON results
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw quantitative results to: {output_json}")

    # 5. Generate Comprehensive Visual Comparison
    pct_axis = [int(round(lvl * 100)) for lvl in fault_levels]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    styles = {
        "multiplicative_bias": {"color": "#d95f02", "marker": "o", "label": "Multiplicative Bias"},
        "gaussian_noise": {"color": "#7570b3", "marker": "s", "label": "Gaussian Noise"},
        "sensor_dropout": {"color": "#e7298a", "marker": "^", "label": "Sensor Dropout"},
    }

    # (1) Accuracy Degradation
    ax = axes[0, 0]
    for ft, s in styles.items():
        accs = [a * 100 for a in results[ft]["accuracy"]]
        ax.plot(pct_axis, accs, marker=s["marker"], color=s["color"], label=s["label"], linewidth=2.2)
    ax.set_title("Classification Accuracy vs Fault Severity", fontweight="bold")
    ax.set_xlabel("Fault Severity (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # (2) Epistemic Uncertainty Response
    ax = axes[0, 1]
    for ft, s in styles.items():
        ax.plot(pct_axis, results[ft]["mean_uncertainty"], marker=s["marker"], color=s["color"], label=s["label"], linewidth=2.2)
    ax.set_title("Epistemic Uncertainty Response (Vacuity u)", fontweight="bold")
    ax.set_xlabel("Fault Severity (%)")
    ax.set_ylabel("Mean Epistemic Uncertainty u in [0, 1]")
    ax.set_ylim(0.2, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # (3) Macro F1-Score
    ax = axes[1, 0]
    for ft, s in styles.items():
        ax.plot(pct_axis, results[ft]["macro_f1"], marker=s["marker"], color=s["color"], label=s["label"], linewidth=2.2)
    ax.set_title("Macro F1-Score vs Fault Severity", fontweight="bold")
    ax.set_xlabel("Fault Severity (%)")
    ax.set_ylabel("Macro F1-Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # (4) Confidence vs Uncertainty Tradeoff
    ax = axes[1, 1]
    for ft, s in styles.items():
        ax.plot(results[ft]["mean_uncertainty"], [a * 100 for a in results[ft]["accuracy"]], marker=s["marker"], color=s["color"], label=s["label"], linewidth=2.2)
    ax.set_title("Accuracy vs Uncertainty Correlation", fontweight="bold")
    ax.set_xlabel("Mean Epistemic Uncertainty")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Saved comparative evaluation plots to: {output_plot}")

    print("\n" + "=" * 80)
    print("  Fault robustness analysis completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EDL robustness across faulty test sets")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/edl_husky.pth")
    parser.add_argument("--test-meta", type=str, default="checkpoints/edl_test_data.pt")
    parser.add_argument("--output-plot", type=str, default="fault_robustness_analysis.png")
    parser.add_argument("--output-json", type=str, default="fault_robustness_results.json")
    args = parser.parse_args()

    evaluate_fault_sweep(
        checkpoint_path=args.checkpoint,
        test_meta_path=args.test_meta,
        output_plot=args.output_plot,
        output_json=args.output_json,
    )
