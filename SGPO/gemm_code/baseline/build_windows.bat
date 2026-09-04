@echo off
setlocal

set "VCVARS64=D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS64%" (
  call "%VCVARS64%"
)

if not exist build mkdir build

nvcc -O3 -std=c++17 cublas_sgemm_fp32.cu -lcublas -o build\cublas_sgemm_fp32.exe
if errorlevel 1 exit /b %errorlevel%

nvcc -O3 -std=c++17 cublas_sgemm_tf32.cu -lcublas -o build\cublas_sgemm_tf32.exe
if errorlevel 1 exit /b %errorlevel%

nvcc -O3 -std=c++17 cublaslt_matmul_tf32.cu -lcublas -lcublasLt -o build\cublaslt_matmul_tf32.exe
if errorlevel 1 exit /b %errorlevel%

echo Baseline build complete.
