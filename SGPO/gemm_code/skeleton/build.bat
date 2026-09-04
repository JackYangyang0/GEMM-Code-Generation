@echo off
setlocal EnableDelayedExpansion

if exist "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
  call "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
  call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)

if "%CUDA_PATH%"=="" (
  echo CUDA_PATH is not set. Please run from a CUDA-enabled developer shell.
  exit /b 1
)

set "SOURCES="
if exist main.cu (
  set "SOURCES=main.cu"
) else if exist main.cpp (
  set "SOURCES=main.cpp"
) else (
  echo missing main.cu or main.cpp
  exit /b 1
)

for %%F in (*.cu) do (
  if /I not "%%F"=="main.cu" set "SOURCES=!SOURCES! %%F"
)

nvcc -O3 -std=c++17 -arch=sm_89 -x cu !SOURCES! -lcublas -o sgpo_gemm.exe

endlocal
