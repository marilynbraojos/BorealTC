from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

CHANNELS = [
    "wx", "wy", "wz",     # Angular velocity (rad/s) from IMU
    "ax", "ay", "az",     # Linear acceleration (m/s^2) from IMU
    "curL", "curR",       # Motor current left/right (A)
    "velL", "velR",       # Wheel velocity left/right (m/s)
]

DEFAULT_TERRAINS = ["asphalt", "flooring", "ice", "sandy_loam", "snow"]


def align_imu_and_pro(imu_df: pd.DataFrame, pro_df: pd.DataFrame) -> pd.DataFrame:
    """Aligns IMU (100 Hz) and proprioceptive (~6.5 Hz) data onto a common 100 Hz time grid."""
    imu_df = imu_df.copy()
    pro_df = pro_df.copy()

    # Convert time in seconds to Timedelta index
    imu_df.index = pd.to_timedelta(imu_df["time"], unit="s")
    pro_df.index = pd.to_timedelta(pro_df["time"], unit="s")

    # Drop the raw time column before resampling/merging
    imu_df = imu_df.drop(columns=["time"])
    pro_df = pro_df.drop(columns=["time"])

    # Resample to 10ms (100 Hz) matching the IMU sampling frequency, forward filling proprioception
    resampled_imu = imu_df.resample("10ms").mean().ffill().bfill()
    resampled_pro = pro_df.resample("10ms").mean().ffill().bfill()

    # Join and ensure exact column order
    fused = pd.concat([resampled_imu, resampled_pro], axis=1).ffill().bfill()
    return fused[CHANNELS]


class HuskyDataset(Dataset):
    """PyTorch Dataset for Clearpath Husky A200 terrain classification data."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/borealtc",
        terrains: Optional[List[str]] = None,
        window_size: int = 100,  # 100 steps @ 100Hz = 1.0 second
        step_size: int = 50,    # 50 steps @ 100Hz = 0.5 second stride (50% overlap)
        channels_first: bool = False,  # If True, shape is (C, T), else (T, C)
        max_runs_per_terrain: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.step_size = step_size
        self.channels_first = channels_first

        # Resolve terrains and mapping
        available_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        dir_lookup = {d.stem.lower(): d for d in available_dirs}

        if terrains is None:
            self.terrains = [t for t in DEFAULT_TERRAINS if t in dir_lookup]
        else:
            self.terrains = [t.lower() for t in terrains]

        self.class_to_idx = {name: idx for idx, name in enumerate(self.terrains)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        self.windows = []  # List of tuples: (tensor_slice, label_idx, terrain_name, run_id)
        self.runs_summary = {}

        for terrain in self.terrains:
            if terrain not in dir_lookup:
                continue
            terrain_dir = dir_lookup[terrain]
            imu_files = sorted(terrain_dir.glob("imu_*.csv"))
            if max_runs_per_terrain:
                imu_files = imu_files[:max_runs_per_terrain]

            self.runs_summary[terrain] = len(imu_files)
            label_idx = self.class_to_idx[terrain]

            for imu_path in imu_files:
                run_id = imu_path.stem.split("_")[1]
                pro_path = terrain_dir / f"pro_{run_id}.csv"

                if not pro_path.exists():
                    continue

                imu_df = pd.read_csv(imu_path)
                pro_df = pd.read_csv(pro_path)
                fused_df = align_imu_and_pro(imu_df, pro_df)

                data_array = fused_df.to_numpy(dtype=np.float32)
                num_steps = len(data_array)

                for start in range(0, num_steps - self.window_size + 1, self.step_size):
                    end = start + self.window_size
                    window_slice = data_array[start:end, :]  # Shape: (T, C)
                    if self.channels_first:
                        window_slice = window_slice.T  # Shape: (C, T)

                    self.windows.append(
                        (
                            torch.from_numpy(window_slice),
                            label_idx,
                            terrain,
                            run_id,
                        )
                    )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        data_tensor, label, terrain, run_id = self.windows[idx]
        return {
            "data": data_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "terrain": terrain,
            "run_id": run_id,
        }


def get_husky_dataloader(
    data_dir: Union[str, Path] = "data/borealtc",
    terrains: Optional[List[str]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    window_size: int = 100,
    step_size: int = 50,
    channels_first: bool = False,
    num_workers: int = 0,
    max_runs_per_terrain: Optional[int] = None,
) -> Tuple[DataLoader, HuskyDataset]:
    """Factory function to quickly instantiate a HuskyDataset and PyTorch DataLoader."""
    dataset = HuskyDataset(
        data_dir=data_dir,
        terrains=terrains,
        window_size=window_size,
        step_size=step_size,
        channels_first=channels_first,
        max_runs_per_terrain=max_runs_per_terrain,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader, dataset
