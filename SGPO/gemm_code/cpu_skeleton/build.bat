@echo off
setlocal
if not exist build mkdir build

where cl >nul 2>nul
if %errorlevel%==0 (
  cl /O2 /TC main.c cpu_kernel.c /Fe:build\gemm_cpu.exe
  exit /b %errorlevel%
)

where gcc >nul 2>nul
if %errorlevel%==0 (
  gcc -O3 -std=c11 main.c cpu_kernel.c -lm -o build\gemm_cpu.exe
  exit /b %errorlevel%
)

echo No C compiler found. Install MSVC Build Tools, gcc, or clang.
exit /b 1
