# SGPO GEMM Baselines

This folder keeps separate baseline programs so SGPO results are not compared against a single ambiguous "cuBLAS" number.

## CUDA Baselines

- `cublas_sgemm_fp32.cu`
  - Uses `cublasSgemm`
  - Uses `CUBLAS_DEFAULT_MATH`
  - Intended as traditional FP32 SGEMM baseline

- `cublas_sgemm_tf32.cu`
  - Uses `cublasSgemm`
  - Requests `CUBLAS_TF32_TENSOR_OP_MATH` when available
  - Useful on Ampere-or-newer GPUs

- `cublaslt_matmul_tf32.cu`
  - Uses `cublasLtMatmul`
  - Uses row-major matrix layouts
  - Requests `CUBLAS_COMPUTE_32F_FAST_TF32`
  - Closer to the high-throughput path used by modern frameworks

All CUDA programs use CUDA events with warmup and device synchronization.

## Torch Baseline

- `torch_matmul_baseline.py`
  - Runs `A @ B`
  - Measures both `allow_tf32=False` and `allow_tf32=True`
  - Uses CUDA events and `torch.cuda.synchronize()`

## Build

Windows:

```bat
cd D:\Paper-Code\GEMM-Code-Generation\SGPO\gemm_code\baseline
build_windows.bat
```

Linux:

```bash
cd /root/scope/gemm_code/baseline
bash build_linux.sh
```

## Run

Arguments are:

```text
M K N [iters] [warmup]
```

Examples:

```bash
./build/cublas_sgemm_fp32 1024 1024 1024 100 10
./build/cublas_sgemm_tf32 1024 1024 1024 100 10
./build/cublaslt_matmul_tf32 1024 1024 1024 100 10
python torch_matmul_baseline.py 1024 1024 1024 --iters 100 --warmup 10
```

Each program prints a `SGPO_BASELINE` line with `latency_ms` and `gflops`.
