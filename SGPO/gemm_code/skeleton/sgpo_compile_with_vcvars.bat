@echo off
call "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.EXE" -O3 -std=c++17 -arch=sm_89 -x cu main.cpp -lcublas -o D:\Paper-Code\GEMM-Code-Generation\SGPO\gemm_code\skeleton\build\gemm.exe --ptxas-options=-v
exit /b %errorlevel%
