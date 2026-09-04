from __future__ import annotations

import argparse
import json
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch CUDA matmul baseline with explicit synchronization.")
    parser.add_argument("M", type=int, nargs="?", default=512)
    parser.add_argument("K", type=int, nargs="?", default=512)
    parser.add_argument("N", type=int, nargs="?", default=512)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def run_once(M: int, K: int, N: int, warmup: int, iters: int, allow_tf32: bool) -> dict[str, object]:
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    device = torch.device("cuda")
    A = ((torch.arange(M * K, device=device, dtype=torch.float32) % 17) - 8).reshape(M, K) / 17.0
    B = ((torch.arange(K * N, device=device, dtype=torch.float32) % 13) - 6).reshape(K, N) / 13.0
    C = torch.empty((M, N), device=device, dtype=torch.float32)

    for _ in range(warmup):
        C = A @ B
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        C = A @ B
    stop_event.record()
    torch.cuda.synchronize()
    event_latency_ms = start_event.elapsed_time(stop_event) / iters

    start = time.perf_counter()
    for _ in range(iters):
        C = A @ B
    torch.cuda.synchronize()
    wall_latency_ms = (time.perf_counter() - start) * 1000.0 / iters

    flops = 2.0 * M * N * K
    gflops = (flops * 1.0e-9) / (event_latency_ms / 1000.0)
    return {
        "name": "torch_matmul",
        "math_mode": "tf32_allowed" if allow_tf32 else "fp32_no_tf32",
        "M": M,
        "K": K,
        "N": N,
        "iters": iters,
        "warmup": warmup,
        "latency_ms": event_latency_ms,
        "wall_latency_ms": wall_latency_ms,
        "gflops": gflops,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for torch baseline.")
    for allow_tf32 in (False, True):
        result = run_once(args.M, args.K, args.N, args.warmup, args.iters, allow_tf32)
        print(json.dumps(result, ensure_ascii=False))
        print(
            "SGPO_BASELINE "
            f"name={result['name']} "
            f"math_mode={result['math_mode']} "
            f"M={result['M']} K={result['K']} N={result['N']} "
            f"latency_ms={result['latency_ms']:.9f} "
            f"gflops={result['gflops']:.9f}"
        )


if __name__ == "__main__":
    main()
