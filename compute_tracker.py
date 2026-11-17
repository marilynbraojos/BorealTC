"""
Compute Tracking and Profiling Module for Husky A200 Terrain Classification Models.
Measures:
1. Model Parameter Complexity (Total, Trainable, Non-trainable, Memory Size)
2. Computational Complexity: Multiply-Accumulate Operations (MACs) and FLOPs
3. Latency & Real-Time Performance: Mean, Median (P50), P95, P99, Min, Max Latency
4. Throughput (Samples / Windows processed per second)
5. Real-Time Factor (RTF = inference latency / sensor window duration)
6. Peak RAM & GPU Memory Allocation
"""

import json
import os
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Counts total, trainable, and non-trainable parameters, and model weight size."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate buffer memory (e.g. means, covariance matrices)
    total_buffers = sum(b.numel() for b in model.buffers())
    
    # Model size in KB and MB (assuming float32: 4 bytes per parameter)
    param_bytes = (total_params + total_buffers) * 4
    size_kb = param_bytes / 1024.0
    size_mb = size_kb / 1024.0

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "buffer_elements": total_buffers,
        "model_size_kb": round(size_kb, 2),
        "model_size_mb": round(size_mb, 4),
    }


def estimate_conv1d_macs(
    in_channels: int, out_channels: int, kernel_size: int, out_length: int
) -> int:
    """Calculates Multiply-Accumulate Operations (MACs) for a 1D Conv layer."""
    return in_channels * out_channels * kernel_size * out_length


def estimate_model_macs(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (1, 10, 100),
) -> Tuple[int, int]:
    """
    Dynamically computes exact MACs and FLOPs for any 1D CNN or Deep Ensemble
    using forward hooks across all Conv1d and Linear layers.
    1 MAC = 2 FLOPs (1 multiply + 1 accumulate).
    """
    device = next(model.parameters()).device
    macs_list = []

    def conv1d_hook(module: nn.Conv1d, inp, out):
        batch_size, out_channels, out_len = out.shape
        kernel_ops = module.in_channels * module.kernel_size[0] // module.groups
        macs = batch_size * out_channels * out_len * kernel_ops
        macs_list.append(macs)

    def linear_hook(module: nn.Linear, inp, out):
        batch_size = out.shape[0]
        macs = batch_size * module.in_features * module.out_features
        macs_list.append(macs)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(input_shape, device=device)
        _ = model(dummy)

    for h in hooks:
        h.remove()

    if was_training:
        model.train()

    total_macs = sum(macs_list)

    # Account for Mahalanobis density gate if initialized
    if hasattr(model, "density_initialized") and bool(model.density_initialized.item()):
        b = input_shape[0]
        total_macs += b * (128 * 128 + 128 * 5)

    total_flops = 2 * total_macs
    return int(total_macs), int(total_flops)


def profile_latency_and_throughput(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (1, 10, 100),
    device: Optional[torch.device] = None,
    warmup_runs: int = 50,
    test_runs: int = 300,
    window_duration_sec: float = 1.0,
) -> Dict[str, float]:
    """
    High-precision latency and throughput benchmark using warmup and percentile statistics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    sample_input = torch.randn(input_shape, device=device)

    # 1. Warm-up runs to stabilize CPU/GPU caches
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)

    # Synchronize if on GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 2. Benchmark runs
    latencies_ms = []
    with torch.no_grad():
        for _ in range(test_runs):
            start = time.perf_counter()
            _ = model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies_ms))
    median_lat = float(np.median(latencies_ms))
    std_lat = float(np.std(latencies_ms))
    p90_lat = float(np.percentile(latencies_ms, 90))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))

    # Throughput (windows per second)
    batch_size = input_shape[0]
    throughput_windows_sec = (1000.0 / mean_lat) * batch_size

    # Real-Time Factor (RTF = mean_latency_sec / window_duration_sec)
    # RTF < 1.0 means faster than real-time; 0.001 means 1000x faster than real-time
    mean_lat_sec = mean_lat / 1000.0
    real_time_factor = mean_lat_sec / window_duration_sec
    real_time_speedup = 1.0 / (real_time_factor + 1e-9)

    return {
        "mean_latency_ms": round(mean_lat, 4),
        "median_latency_p50_ms": round(median_lat, 4),
        "std_latency_ms": round(std_lat, 4),
        "p90_latency_ms": round(p90_lat, 4),
        "p95_latency_ms": round(p95_lat, 4),
        "p99_latency_ms": round(p99_lat, 4),
        "min_latency_ms": round(min_lat, 4),
        "max_latency_ms": round(max_lat, 4),
        "throughput_windows_per_sec": round(throughput_windows_sec, 2),
        "real_time_factor_rtf": round(real_time_factor, 6),
        "real_time_speedup": round(real_time_speedup, 1),
    }


def measure_memory_footprint(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (1, 10, 100),
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Measures peak RAM and VRAM allocation during a forward pass."""
    if device is None:
        device = next(model.parameters()).device

    tracemalloc.start()
    sample_input = torch.randn(input_shape, device=device)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = model(sample_input)

    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_ram_mb = peak_ram / (1024.0 * 1024.0)

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    return {
        "peak_ram_mb": round(peak_ram_mb, 4),
        "peak_vram_mb": round(peak_vram_mb, 4),
    }


class ComputeTracker:
    """
    Unified manager for tracking, profiling, and persistent logging of model compute metrics.
    """

    def __init__(
        self,
        model_name: str = "FineTuned_EDL_1DCNN",
        output_dir: str = "metrics",
        input_shape: Tuple[int, int, int] = (1, 10, 100),
        window_duration_sec: float = 1.0,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_shape = input_shape
        self.window_duration_sec = window_duration_sec

    def benchmark_model(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        test_runs: int = 300,
        save_log: bool = True,
    ) -> Dict[str, Union[str, int, float, dict]]:
        """Executes full compute profiling suite on model."""
        if device is None:
            device = next(model.parameters()).device

        # 1. Parameter statistics
        param_stats = count_parameters(model)

        # 2. FLOPs and MACs
        macs, flops = estimate_model_macs(model, self.input_shape)
        flops_stats = {
            "macs_per_window": macs,
            "macs_m": round(macs / 1e6, 4),
            "flops_per_window": flops,
            "mflops_per_window": round(flops / 1e6, 4),
            "gflops_per_window": round(flops / 1e9, 6),
        }

        # 3. Latency & Throughput
        lat_stats = profile_latency_and_throughput(
            model=model,
            input_shape=self.input_shape,
            device=device,
            test_runs=test_runs,
            window_duration_sec=self.window_duration_sec,
        )

        # 4. Memory footprint
        mem_stats = measure_memory_footprint(model, self.input_shape, device)

        # 5. Composite benchmark record
        record = {
            "model_name": self.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "input_shape": list(self.input_shape),
            "window_duration_sec": self.window_duration_sec,
            "parameters": param_stats,
            "computational_complexity": flops_stats,
            "latency_metrics": lat_stats,
            "memory_footprint": mem_stats,
        }

        if save_log:
            self.save_benchmark(record)

        return record

    def save_benchmark(self, record: dict):
        """Saves compute record as formatted JSON and appends to CSV summary table."""
        json_path = self.output_dir / f"{self.model_name}_compute_benchmark.json"
        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)

        csv_path = self.output_dir / "compute_benchmarks_summary.csv"
        file_exists = csv_path.exists()

        p = record["parameters"]
        c = record["computational_complexity"]
        l = record["latency_metrics"]
        m = record["memory_footprint"]

        header = (
            "timestamp,model_name,device,total_params,size_kb,mflops,mean_lat_ms,"
            "p95_lat_ms,throughput_fps,rtf,speedup,peak_ram_mb\n"
        )
        row = (
            f"{record['timestamp']},{record['model_name']},{record['device']},"
            f"{p['total_parameters']},{p['model_size_kb']},{c['mflops_per_window']},"
            f"{l['mean_latency_ms']},{l['p95_latency_ms']},{l['throughput_windows_per_sec']},"
            f"{l['real_time_factor_rtf']},{l['real_time_speedup']},{m['peak_ram_mb']}\n"
        )

        with open(csv_path, "a") as f:
            if not file_exists:
                f.write(header)
            f.write(row)

        print(f"Logged compute benchmark to: {json_path}")
        print(f"Updated cumulative summary: {csv_path}")

    @staticmethod
    def print_summary(record: dict):
        """Displays formatted compute benchmark table in console."""
        p = record["parameters"]
        c = record["computational_complexity"]
        l = record["latency_metrics"]
        m = record["memory_footprint"]

        print("\n" + "=" * 78)
        print(f"  COMPUTE & LATENCY BENCHMARK: {record['model_name']}")
        print("=" * 78)
        print(" [1] PARAMETERS & COMPLEXITY:")
        print(f"  • Total Parameters          : {p['total_parameters']:,} ({p['trainable_parameters']:,} trainable)")
        print(f"  • Model Size on Disk        : {p['model_size_kb']:.2f} KB ({p['model_size_mb']:.4f} MB)")
        print(f"  • Multiply-Accumulates (MAC): {c['macs_m']:.3f} M-MACs per 1.0s window")
        print(f"  • Floating Point Ops (FLOPs): {c['mflops_per_window']:.3f} MFLOPs ({c['flops_per_window']:,} ops)")
        print("\n [2] INFERENCE LATENCY & THROUGHPUT (Device: " + record['device'] + "):")
        print(f"  • Mean Latency              : {l['mean_latency_ms']:.4f} ms (± {l['std_latency_ms']:.4f} ms)")
        print(f"  • Median Latency (P50)      : {l['median_latency_p50_ms']:.4f} ms")
        print(f"  • 95th Percentile Latency   : {l['p95_latency_ms']:.4f} ms")
        print(f"  • 99th Percentile Latency   : {l['p99_latency_ms']:.4f} ms")
        print(f"  • Throughput                : {l['throughput_windows_per_sec']:,} windows/sec")
        print(f"  • Real-Time Factor (RTF)    : {l['real_time_factor_rtf']:.6f} (Lower is better; <1.0 is real-time)")
        print(f"  • Real-Time Margin          : {l['real_time_speedup']:.1f}x faster than real-time robot stream")
        print("\n [3] MEMORY FOOTPRINT:")
        print(f"  • Peak RAM (Inference)      : {m['peak_ram_mb']:.4f} MB")
        if m.get("peak_vram_mb", 0) > 0:
            print(f"  • Peak VRAM (GPU)           : {m['peak_vram_mb']:.4f} MB")
        print("=" * 78 + "\n")


if __name__ == "__main__":
    from edl_model import FineTunedEvidentialClassifier1D, EvidentialClassifier1D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running compute benchmark on device: {device}")

    # Benchmark Fine-Tuned EDL Model
    ft_model = FineTunedEvidentialClassifier1D(num_classes=5).to(device)
    if Path("checkpoints/edl_husky_finetuned.pth").exists():
        ckpt = torch.load("checkpoints/edl_husky_finetuned.pth", map_location=device, weights_only=False)
        ft_model.load_state_dict(ckpt["model_state_dict"])

    tracker = ComputeTracker(model_name="FineTuned_EDL_1DCNN")
    record = tracker.benchmark_model(ft_model, device=device)
    tracker.print_summary(record)
