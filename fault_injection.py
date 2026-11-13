"""
Fault injection module for robot sensor streams (Husky A200).
Simulates real-world sensor degradation, hardware failure, and communication faults:
1. Multiplicative Bias (sensor calibration drift / scale factor errors)
2. Gaussian Noise (electromagnetic interference / thermal sensor noise)
3. Sensor Dropout (disconnected sensors, dead channels, or packet dropouts)
"""

from typing import Literal, Optional, Union
import torch
import numpy as np


def apply_multiplicative_bias(
    x: torch.Tensor,
    bias_pct: float,
) -> torch.Tensor:
    """
    Applies multiplicative bias to sensor readings: x_faulty = x * (1 + bias_pct).
    
    Args:
        x: Input sensor tensor of shape (batch, channels, time) or (channels, time)
        bias_pct: Multiplicative scaling bias (e.g. 0.10 for +10% scaling bias)
        
    Returns:
        Tensor with multiplicative bias applied.
    """
    if bias_pct == 0.0:
        return x.clone()
    scale = 1.0 + float(bias_pct)
    return x * scale


def apply_gaussian_noise(
    x: torch.Tensor,
    noise_pct: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Adds zero-mean Gaussian noise scaled relative to each channel's empirical std.
    Noise std = noise_pct * std(x_channel).
    
    Args:
        x: Input sensor tensor of shape (batch, channels, time) or (channels, time)
        noise_pct: Relative noise intensity (e.g. 0.20 for noise std = 20% of signal std)
        generator: Optional PyTorch random generator for reproducibility
        
    Returns:
        Tensor corrupted with Gaussian noise.
    """
    if noise_pct == 0.0:
        return x.clone()
    
    # Calculate channel-wise standard deviation along time dimension
    time_dim = -1
    std_per_channel = torch.std(x, dim=time_dim, keepdim=True)
    # Avoid zero std edge cases
    std_per_channel = torch.clamp(std_per_channel, min=1e-5)
    
    noise_sigma = noise_pct * std_per_channel
    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator) * noise_sigma
    return x + noise


def apply_sensor_dropout(
    x: torch.Tensor,
    dropout_pct: float,
    mode: Literal["channel", "packet"] = "channel",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Zeroes out sensor readings to simulate hardware disconnects or communication loss.
    
    Args:
        x: Input tensor of shape (batch, channels, time) or (channels, time)
        dropout_pct: Fraction of channels or packets to drop in [0.0, 1.0] (e.g. 0.3 for 30%)
        mode:
            - "channel": Drops entire sensor channels (e.g. broken IMU axis or wheel encoder)
            - "packet": Drops random time packets across all channels
        generator: Optional PyTorch random generator
        
    Returns:
        Tensor with dropped values set to 0.0.
    """
    if dropout_pct == 0.0:
        return x.clone()
    
    x_faulty = x.clone()
    is_batched = (x.ndim == 3)
    
    if mode == "channel":
        channels = x.shape[1] if is_batched else x.shape[0]
        # Number of channels to zero out
        num_to_drop = int(round(dropout_pct * channels))
        if num_to_drop > 0:
            if is_batched:
                batch_size = x.shape[0]
                for b in range(batch_size):
                    drop_indices = torch.randperm(channels, generator=generator)[:num_to_drop]
                    x_faulty[b, drop_indices, :] = 0.0
            else:
                drop_indices = torch.randperm(channels, generator=generator)[:num_to_drop]
                x_faulty[drop_indices, :] = 0.0
    elif mode == "packet":
        # Random binary mask along the time dimension
        mask = torch.rand(x.shape, generator=generator, device=x.device) >= dropout_pct
        x_faulty = x_faulty * mask.to(x.dtype)
    else:
        raise ValueError(f"Unknown dropout mode: {mode}")

    return x_faulty
