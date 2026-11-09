#!/usr/bin/env python3
"""
Simple script to load and inspect Clearpath Husky A200 data using PyTorch DataLoader.
"""

from husky_dataloader import get_husky_dataloader, CHANNELS


def main():
    print("=" * 70)
    print("  Loading Husky A200 Terrain Data via PyTorch DataLoader")
    print("=" * 70)

    # 1. Instantiate dataloader
    # window_size=100 (1.0 second @ 100Hz), step_size=50 (0.5 second stride)
    # Using max_runs_per_terrain=2 for quick demonstration, or set to None for all runs
    batch_size = 16
    dataloader, dataset = get_husky_dataloader(
        data_dir="data/borealtc",
        terrains=["asphalt", "flooring", "ice", "sandy_loam", "snow"],
        batch_size=batch_size,
        shuffle=True,
        window_size=100,
        step_size=50,
        max_runs_per_terrain=2,
    )

    # 2. Print Dataset Overview
    print("\n[1] Dataset Summary:")
    print(f"  • Data directory : {dataset.data_dir}")
    print(f"  • Classes/Labels : {dataset.class_to_idx}")
    print("  • Runs loaded    :")
    for terrain, count in dataset.runs_summary.items():
        print(f"      - {terrain:<12}: {count} runs")
    print(f"  • Total windows  : {len(dataset):,} samples")
    print(f"  • Window length  : {dataset.window_size} timesteps (1.0s @ 100Hz)")
    print(f"  • Channels ({len(CHANNELS)})   : {CHANNELS}")

    # 3. Fetch a single batch from the DataLoader
    print(f"\n[2] Fetching a batch (batch_size={batch_size})...")
    batch = next(iter(dataloader))
    data = batch["data"]          # Tensor shape: [batch_size, window_size, 10]
    labels = batch["label"]       # Tensor shape: [batch_size]
    terrains = batch["terrain"]   # List of class names
    run_ids = batch["run_id"]     # List of run IDs

    print(f"  • Batch Data Tensor Shape  : {list(data.shape)}  [batch_size, time_steps, channels]")
    print(f"  • Batch Labels Tensor Shape: {list(labels.shape)}")
    print(f"  • Labels in this batch     : {labels.tolist()}")
    print(f"  • Terrains in this batch   : {list(terrains)}")
    print(f"  • Run IDs                  : {list(run_ids)}")

    # 4. Detailed view of Sample #0 in the batch
    print("\n[3] Inspecting Sample #0 in the batch:")
    sample_terrain = terrains[0]
    sample_label = labels[0].item()
    sample_run = run_ids[0]
    sample_tensor = data[0]  # Shape: [100, 10]

    print(f"  • Terrain : {sample_terrain.upper()} (Class idx: {sample_label}, Run: {sample_run})")
    print(f"  • Tensor  : shape={sample_tensor.shape}, dtype={sample_tensor.dtype}")
    print("\n  First 5 timesteps (t = 0.00s to 0.04s):")
    header = f"  {'Step':<5} | {'wx':<8} {'wy':<8} {'wz':<8} | {'ax':<8} {'ay':<8} {'az':<8} | {'curL':<7} {'curR':<7} | {'velL':<7} {'velR':<7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for step in range(5):
        vals = sample_tensor[step].tolist()
        wx, wy, wz, ax, ay, az, curL, curR, velL, velR = vals
        print(
            f"  {step:<5} | {wx:8.4f} {wy:8.4f} {wz:8.4f} | "
            f"{ax:8.3f} {ay:8.3f} {az:8.3f} | "
            f"{curL:7.2f} {curR:7.2f} | "
            f"{velL:7.3f} {velR:7.3f}"
        )

    # 5. Channel statistics for this window
    print("\n  Summary Statistics for Sample #0 across all 100 timesteps:")
    print(f"  {'Channel':<8} | {'Mean':>10} | {'Min':>10} | {'Max':>10} | Description")
    print("  " + "-" * 65)

    descriptions = {
        "wx": "Angular vel X (rad/s)",
        "wy": "Angular vel Y (rad/s)",
        "wz": "Angular vel Z (rad/s)",
        "ax": "Linear acc X (m/s^2)",
        "ay": "Linear acc Y (m/s^2)",
        "az": "Linear acc Z (m/s^2)",
        "curL": "Motor current Left (A)",
        "curR": "Motor current Right (A)",
        "velL": "Wheel speed Left (m/s)",
        "velR": "Wheel speed Right (m/s)",
    }

    for col_idx, ch_name in enumerate(CHANNELS):
        ch_vals = sample_tensor[:, col_idx]
        mean_val = ch_vals.mean().item()
        min_val = ch_vals.min().item()
        max_val = ch_vals.max().item()
        desc = descriptions.get(ch_name, "")
        print(f"  {ch_name:<8} | {mean_val:10.4f} | {min_val:10.4f} | {max_val:10.4f} | {desc}")

    print("\n" + "=" * 70)
    print("  Data loader test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
