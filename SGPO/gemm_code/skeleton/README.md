# SGPO GEMM Skeleton

This directory is the patch target for SGPO-generated CUDA GEMM code.

Files:

- `main.cpp`: benchmark and correctness harness.
- `kernel.h`: stable public `cuda_gemm` declaration.
- `cuda_kernel.cuh`: CUDA kernel template and kernel-body patch regions.

Optional generated CUDA source files such as `gemm_kernel.cu` are compiled
automatically when they are present in this directory.
Header-only CUDA code in `cuda_kernel.cuh` is included through `kernel.h`;
do not pass `.cuh` files to `nvcc` as standalone source files.

Patch anchors:

- `SGPO_PATCH_LAUNCH_CONFIG_BEGIN/END`
- `SGPO_PATCH_KERNEL_LAUNCH_BEGIN/END`
- `SGPO_PATCH_SHARED_DECL_BEGIN/END`
- `SGPO_PATCH_MAIN_LOOP_BEGIN/END`
- `SGPO_PATCH_STORE_BEGIN/END`

Build:

```bat
build.bat
sgpo_gemm.exe 512 512 512
```

Linux build:

```bash
chmod +x build_linux.sh
./build_linux.sh 512 512 512
```

The script requests `sm_89` by default. If the installed `nvcc` does not
support `sm_89` but supports `sm_87`, it automatically falls back to `sm_87`.

Equivalent Linux command:

```bash
mkdir -p build
nvcc -O3 -std=c++17 -arch=sm_89 -x cu main.cpp -lcublas -o build/sgpo_gemm
./build/sgpo_gemm 512 512 512
```

If a generated implementation later creates an extra `.cu` source file, add it explicitly:

```bash
nvcc -O3 -std=c++17 -arch=sm_89 -x cu main.cpp gemm_kernel.cu -lcublas -o build/sgpo_gemm
```

Override architecture:

```bash
ARCH=sm_86 ./build_linux.sh 1024 1024 1024
```
