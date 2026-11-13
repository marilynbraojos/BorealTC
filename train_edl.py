#!/usr/bin/env python3
"""
Training script for Evidential Deep Learning (EDL) on Husky A200 terrain data.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from husky_dataloader import HuskyDataset
from edl_model import EvidentialClassifier1D, edl_loss


def train(
    data_dir: str = "data/borealtc",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "checkpoints/edl_husky.pth",
    test_save_path: str = "checkpoints/edl_test_data.pt",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset with channels_first=True for 1D CNN: shape (10, 100)
    print(f"Loading Husky A200 data from {data_dir}...")
    dataset = HuskyDataset(
        data_dir=data_dir,
        window_size=100,
        step_size=50,
        channels_first=True,
    )
    num_samples = len(dataset)
    num_classes = len(dataset.terrains)
    print(f"Loaded {num_samples} total window samples across {num_classes} terrain classes:")
    for k, v in dataset.class_to_idx.items():
        print(f"  - Class {v}: {k}")

    # 2. Stratified Train / Test split (80% train, 20% test)
    all_labels = [sample[1] for sample in dataset.windows]
    train_indices, test_indices = train_test_split(
        np.arange(num_samples),
        test_size=0.20,
        random_state=seed,
        stratify=all_labels,
    )

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print(f"Dataset split: {len(train_indices)} train samples, {len(test_indices)} test samples.")

    # 3. Initialize Model and Optimizer
    model = EvidentialClassifier1D(in_channels=10, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Training Loop
    print("\nStarting Evidential Deep Learning training...")
    print("-" * 75)
    print(f"{'Epoch':<7} | {'Train Loss':<11} | {'ACE Loss':<10} | {'KL Loss':<10} | {'Train Acc':<10} | {'Avg Uncert':<10}")
    print("-" * 75)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_accum = 0.0
        ace_loss_accum = 0.0
        kl_loss_accum = 0.0
        correct = 0
        total = 0
        uncertainty_accum = 0.0

        for batch in train_loader:
            inputs = batch["data"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad()
            evidence = model(inputs)

            loss, ace_l, kl_l = edl_loss(
                evidence=evidence,
                targets=targets,
                epoch=epoch,
                num_classes=num_classes,
                annealing_epochs=10,
            )

            loss.backward()
            optimizer.step()

            # Tracking metrics
            total_loss_accum += loss.item() * len(targets)
            ace_loss_accum += ace_l.item() * len(targets)
            kl_loss_accum += kl_l.item() * len(targets)

            # Evidential predictions & uncertainty
            with torch.no_grad():
                alpha = evidence + 1.0
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                preds = torch.argmax(probs, dim=1)
                uncert = num_classes / S.squeeze(1)

                correct += (preds == targets).sum().item()
                total += len(targets)
                uncertainty_accum += torch.sum(uncert).item()

        scheduler.step()

        epoch_loss = total_loss_accum / total
        epoch_ace = ace_loss_accum / total
        epoch_kl = kl_loss_accum / total
        epoch_acc = correct / total
        epoch_uncert = uncertainty_accum / total

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"{epoch:<7} | {epoch_loss:<11.4f} | {epoch_ace:<10.4f} | "
                f"{epoch_kl:<10.4f} | {epoch_acc * 100:<9.2f}% | {epoch_uncert:<10.4f}"
            )

    print("-" * 75)

    # 5. Save model checkpoint and test dataset indices
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "in_channels": 10,
        "num_classes": num_classes,
        "class_to_idx": dataset.class_to_idx,
        "terrains": dataset.terrains,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved EDL model checkpoint to: {save_path}")

    # Save test partition info for evaluation script
    test_meta = {
        "test_indices": test_indices.tolist(),
        "train_indices": train_indices.tolist(),
        "class_to_idx": dataset.class_to_idx,
        "terrains": dataset.terrains,
        "data_dir": str(data_dir),
    }
    torch.save(test_meta, test_save_path)
    print(f"Saved test split metadata to: {test_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EDL model on Husky A200 terrain data")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="data/borealtc", help="Path to dataset directory")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
