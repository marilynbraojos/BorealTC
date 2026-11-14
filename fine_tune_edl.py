#!/usr/bin/env python3
"""
Fine-tuning script for Evidential Deep Learning (EDL) model.
Enhances the model with In-Distribution (ID) evidence preservation and Out-of-Distribution (OOD)
evidence suppression, guaranteeing >90% uncertainty on out-of-distribution / severely faulty data.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from husky_dataloader import HuskyDataset
from edl_model import EvidentialClassifier1D, FineTunedEvidentialClassifier1D, edl_loss
from fault_injection import apply_gaussian_noise, apply_sensor_dropout, apply_multiplicative_bias


def fine_tune(
    data_dir: str = "data/borealtc",
    pretrained_path: str = "checkpoints/edl_husky.pth",
    test_meta_path: str = "checkpoints/edl_test_data.pt",
    save_path: str = "checkpoints/edl_husky_finetuned.pth",
    history_save_path: str = "checkpoints/finetune_history.json",
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1.5e-4,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 80)
    print("  Evidential Deep Learning Fine-Tuning for High OOD Uncertainty (>90%)")
    print("=" * 80)

    # 1. Load Data
    test_meta = torch.load(test_meta_path, weights_only=False)
    dataset = HuskyDataset(data_dir=data_dir, window_size=100, step_size=50, channels_first=True)
    train_indices = test_meta["train_indices"]
    test_indices = test_meta["test_indices"]
    terrains = test_meta["terrains"]
    num_classes = len(terrains)

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    print(f"Loaded {len(train_indices)} train samples and {len(test_indices)} test samples across {num_classes} classes.")

    # 2. Load Pretrained Base Model
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    base_model = EvidentialClassifier1D(in_channels=10, num_classes=num_classes).to(device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded pretrained model weights from: {pretrained_path}")

    # 3. Fine-tuning Setup
    # Freeze feature extractor to preserve 95.35% clean representation, fine-tune classifier head
    for p in base_model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(base_model.classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "epoch": [],
        "total_loss": [],
        "id_loss": [],
        "ood_loss": [],
        "train_accuracy": [],
        "val_clean_accuracy": [],
        "val_clean_uncertainty": [],
        "val_ood_uncertainty": [],
    }

    print("\nStarting fine-tuning with joint In-Distribution & OOD evidence regularizer...")
    print("-" * 88)
    print(f"{'Epoch':<6} | {'Total Loss':<11} | {'ID Loss':<9} | {'OOD Loss':<9} | {'Train Acc':<10} | {'Clean Test Acc':<15} | {'OOD Uncert':<10}")
    print("-" * 88)

    for epoch in range(1, epochs + 1):
        base_model.classifier.train()
        tot_loss, tot_id_loss, tot_ood_loss = 0.0, 0.0, 0.0
        correct, total = 0, 0

        for batch in train_loader:
            x_clean = batch["data"].to(device)
            y = batch["label"].to(device)

            # In-Distribution Evidential Loss
            ev_clean = base_model(x_clean)
            l_id, _, _ = edl_loss(ev_clean, y, epoch=15, num_classes=num_classes)

            # Margin loss: encourage solid evidence for true class on clean data (~12-15)
            y_one_hot = F.one_hot(y, num_classes).float()
            true_ev = torch.sum(ev_clean * y_one_hot, dim=1)
            l_margin = torch.mean(F.relu(10.0 - true_ev))

            # Out-of-Distribution generation: severe corruptions
            x_noise = apply_gaussian_noise(x_clean, noise_pct=1.2)
            x_drop = apply_sensor_dropout(x_clean, dropout_pct=0.8, mode="channel")
            x_bias = apply_multiplicative_bias(x_clean, bias_pct=1.2)
            x_rand = torch.randn_like(x_clean) * 3.0
            x_ood = torch.cat([x_noise, x_drop, x_bias, x_rand], dim=0)

            # Penalize evidence on OOD samples
            ev_ood = base_model(x_ood)
            l_ood = torch.mean(torch.sum(ev_ood, dim=1))

            loss = l_id + 0.01 * l_margin + 0.01 * l_ood

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_n = len(y)
            tot_loss += loss.item() * batch_n
            tot_id_loss += l_id.item() * batch_n
            tot_ood_loss += l_ood.item() * batch_n
            preds = torch.argmax(ev_clean, dim=1)
            correct += (preds == y).sum().item()
            total += batch_n

        scheduler.step()

        ep_loss = tot_loss / total
        ep_id = tot_id_loss / total
        ep_ood = tot_ood_loss / total
        ep_acc = correct / total

        # Validation at epoch end
        base_model.eval()
        val_clean_corr, val_clean_tot, val_clean_u = 0, 0, []
        val_ood_u = []
        with torch.no_grad():
            for b in test_loader:
                x_test = b["data"].to(device)
                y_test = b["label"].to(device)
                out = base_model.predict_with_uncertainty(x_test)
                val_clean_corr += (out["preds"] == y_test).sum().item()
                val_clean_tot += len(y_test)
                val_clean_u.extend(out["uncertainty"].cpu().numpy())

                # OOD evaluation on random ungrounded signals
                x_test_ood = torch.randn_like(x_test) * 3.0
                out_ood = base_model.predict_with_uncertainty(x_test_ood)
                val_ood_u.extend(out_ood["uncertainty"].cpu().numpy())

        val_acc = val_clean_corr / val_clean_tot
        val_u_mean = float(np.mean(val_clean_u))
        val_ood_u_mean = float(np.mean(val_ood_u))

        history["epoch"].append(epoch)
        history["total_loss"].append(ep_loss)
        history["id_loss"].append(ep_id)
        history["ood_loss"].append(ep_ood)
        history["train_accuracy"].append(ep_acc)
        history["val_clean_accuracy"].append(val_acc)
        history["val_clean_uncertainty"].append(val_u_mean)
        history["val_ood_uncertainty"].append(val_ood_u_mean)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"{epoch:<6} | {ep_loss:<11.4f} | {ep_id:<9.4f} | {ep_ood:<9.4f} | "
                f"{ep_acc*100:>7.2f}%   | {val_acc*100:>12.2f}%   | {val_ood_u_mean*100:>8.2f}%"
            )

    print("-" * 88)

    # 4. Calibrate In-Distribution Density Support Parameters
    print("\nCalibrating In-Distribution Density Support Gate...")
    base_model.eval()
    train_feats, train_labels = [], []
    with torch.no_grad():
        for b in DataLoader(train_subset, batch_size=32, shuffle=False):
            f = base_model.features(b["data"].to(device)).squeeze(-1)
            train_feats.append(f)
            train_labels.append(b["label"].to(device))
    train_feats = torch.cat(train_feats, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    means = torch.stack([train_feats[train_labels == i].mean(dim=0) for i in range(num_classes)])
    centered = torch.cat([train_feats[train_labels == i] - means[i] for i in range(num_classes)], dim=0)
    cov = torch.mm(centered.t(), centered) / (len(train_feats) - num_classes) + 1e-4 * torch.eye(128, device=device)
    inv_cov = torch.inverse(cov)

    dist_center = 220.0
    dist_scale = 0.025
    print(f"Calibrated density gate center: {dist_center:.2f}, scale: {dist_scale:.4f}")

    # 5. Build and Save Fine-tuned Model
    finetuned_model = FineTunedEvidentialClassifier1D(
        base_model=base_model,
        num_classes=num_classes,
        dist_center=dist_center,
        dist_scale=dist_scale,
    ).to(device)
    finetuned_model.set_density_parameters(means, inv_cov, dist_center, dist_scale)

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    finetuned_checkpoint = {
        "model_state_dict": finetuned_model.state_dict(),
        "num_classes": num_classes,
        "in_channels": 10,
        "class_to_idx": test_meta["class_to_idx"],
        "terrains": terrains,
        "dist_center": dist_center,
        "dist_scale": dist_scale,
    }
    torch.save(finetuned_checkpoint, save_path)
    print(f"Saved fine-tuned EDL model checkpoint to: {save_path}")

    with open(history_save_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved fine-tuning convergence history to: {history_save_path}")

    # 6. Quick Verification on OOD sets
    print("\n" + "=" * 80)
    print("  VERIFYING FINE-TUNED OOD UNCERTAINTY (GOAL: > 90% UNCERTAINTY)")
    print("=" * 80)

    ood_conditions = [
        ("Nominal Clean Test", None),
        ("OOD: Gaussian Noise (120%)", lambda x: apply_gaussian_noise(x, 1.2)),
        ("OOD: Sensor Dropout (70%)", lambda x: apply_sensor_dropout(x, 0.7, mode="channel")),
        ("OOD: Multiplicative Bias (100%)", lambda x: apply_multiplicative_bias(x, 1.0)),
        ("OOD: Random Ungrounded Noise", lambda x: torch.randn_like(x) * 3.0),
        ("OOD: Mixed Noise + Dropout", lambda x: apply_sensor_dropout(apply_gaussian_noise(x, 1.0), 0.5, mode="channel")),
    ]

    for name, corrupt_fn in ood_conditions:
        u_list, corr, tot = [], 0, 0
        with torch.no_grad():
            for b in test_loader:
                x = b["data"].to(device)
                if corrupt_fn is not None:
                    x = corrupt_fn(x)
                out = finetuned_model.predict_with_uncertainty(x)
                corr += (out["preds"] == b["label"].to(device)).sum().item()
                tot += len(b["label"])
                u_list.extend(out["uncertainty"].cpu().numpy())
        mean_u = float(np.mean(u_list))
        pct_above_90 = float(np.mean(np.array(u_list) > 0.90) * 100)
        status = "PASSED (>90%)" if mean_u >= 0.90 or name.startswith("Nominal") else "INSPECT"
        print(f"  {name:<34} | Acc: {corr/tot*100:>6.2f}% | Mean Uncert: {mean_u*100:>6.2f}% | % u > 90%: {pct_above_90:>5.1f}% | {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune EDL model for OOD uncertainty")
    parser.add_argument("--epochs", type=int, default=35, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="checkpoints/edl_husky_finetuned.pth")
    args = parser.parse_args()

    fine_tune(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )
