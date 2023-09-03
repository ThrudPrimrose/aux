#!/bin/bash

python benchmark_cuda.py

nvcc --gpu-code=sm_86 --gpu-architecture=compute_86 --generate-line-info --resource-usage --ptxas-options=-v benchmark_cuda_A_mul_Bt_full.cu -o A_Bt_full_opt

sudo env "PATH=$PATH" ncu --export A_Bt_full_opt_rep --force-overwrite  --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0  --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --set full --import-source no --check-exit-code yes --target-processes all A_Bt_full_opt