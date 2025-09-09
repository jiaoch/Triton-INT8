# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# Copyright (C) Zhejiang Lab(ZJLab) 2025. All Right Reserved.
# FilePath: \undefinedc:\Users\hp\Desktop\triton_bwd33.py
# Release version: 1.0
# Date: 2025-09-01
# Author: FanJH
# LastEditTime: 2025-09-01
# LastEditors: FanJH
# Description: 
#############################################
import torch
import triton
import triton.language as tl
from typing import Tuple
import time


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],   
)
@triton.jit
def int8_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    A_scales_ptr, B_scales_ptr,
    BATCH, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bn, stride_bk,
    stride_cb, stride_cm, stride_cn,
    stride_scale_a_b, stride_scale_am, stride_scale_ak,
    stride_scale_b_b, stride_scale_bn, stride_scale_bk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = tl.multiple_of(pid_m * BLOCK_M + tl.arange(0, BLOCK_M), BLOCK_M)
    offs_n = tl.multiple_of(pid_n * BLOCK_N + tl.arange(0, BLOCK_N), BLOCK_N)
    offs_k = tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptr_base = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am
    b_ptr_base = B_ptr + pid_b * stride_bb + offs_n[None, :] * stride_bn

    for k_iter in range(0, K, BLOCK_K):

        a_ptrs = a_ptr_base + (k_iter + offs_k[None, :]) * stride_ak
        a_q = tl.load(a_ptrs)

        b_ptrs = b_ptr_base + (k_iter + offs_k[:, None]) * stride_bk
        b_q = tl.load(b_ptrs)

        a_scales = tl.load(A_scales_ptr + pid_b * stride_scale_a_b + pid_m * stride_scale_am +  (k_iter// BLOCK_K) * stride_scale_ak) 
        b_scales = tl.load(B_scales_ptr + pid_b * stride_scale_b_b + pid_n * stride_scale_bn +  (k_iter// BLOCK_K) * stride_scale_bk) 

        acc += tl.dot(a_q, b_q) * a_scales * b_scales

    c_ptrs = (C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc)

def triton_gemm(A_int8, B_int8, A_scale, B_scale, out_dtype=torch.bfloat16):
    B, M, K = A_int8.shape
    N = B_int8.shape[1]
    BLOCK_M = BLOCK_N = 128
    BLOCK_K = 128
    C = torch.zeros((B, M, N), device=A_int8.device, dtype=out_dtype)
    A_int8 = A_int8.contiguous()
    B_int8 = B_int8.contiguous()
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        B,
    )

    int8_gemm_kernel[grid](
    A_ptr=A_int8, B_ptr=B_int8, C_ptr=C,
    A_scales_ptr=A_scale, B_scales_ptr=B_scale,
    BATCH=B, M=M, N=N, K=K,
    stride_ab=A_int8.stride(0), stride_am=A_int8.stride(1), stride_ak=A_int8.stride(2),
    stride_bb=B_int8.stride(0), stride_bn=B_int8.stride(1), stride_bk=B_int8.stride(2),
    stride_cb=C.stride(0), stride_cm=C.stride(1), stride_cn=C.stride(2),
    stride_scale_a_b=A_scale.stride(0), stride_scale_am=A_scale.stride(1), stride_scale_ak=A_scale.stride(2),
    stride_scale_b_b=B_scale.stride(0), stride_scale_bn=B_scale.stride(1), stride_scale_bk=B_scale.stride(2),
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    # num_warps=4, num_stages=2
    )
    return C


@triton.jit
def per_block_quant_kernel(
    X_ptr, SCALE_ptr, OUT_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    stride_sb, stride_sm, stride_sn,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    num_m_blocks = M // BLOCK_M
    num_n_blocks = N // BLOCK_N

    b = pid // (num_m_blocks * num_n_blocks)
    rem = pid % (num_m_blocks * num_n_blocks)
    m_block = rem // num_n_blocks
    n_block = rem % num_n_blocks

    offs_m = tl.multiple_of(m_block * BLOCK_M + tl.arange(0, BLOCK_M), BLOCK_M)
    offs_n = tl.multiple_of(n_block * BLOCK_N + tl.arange(0, BLOCK_N), BLOCK_N)

    x_ptrs = (X_ptr + b * stride_b + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
    x = tl.load(x_ptrs) 
    amax = tl.max(tl.abs(x))    
    amax = tl.where(amax > 1e-4, amax, 1e-4)
    scale = 127.0 / amax
    x_scaled = x * scale
    x_q = tl.floor(x_scaled + 0.5).to(tl.int8)

    out_ptrs = (OUT_ptr + b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, x_q)
    scale_inv = 1.0 / scale
    scale_ptr = (SCALE_ptr + b * stride_sb + m_block * stride_sm + n_block * stride_sn)
    tl.store(scale_ptr, scale_inv)

def triton_per_block_quant(x: torch.Tensor):
    B, M, N = x.shape
    BLOCK_M = 128
    BLOCK_N = 128
    x_int8 = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((B, M // BLOCK_M, N // BLOCK_N), dtype=torch.float32, device=x.device)
    grid = (B * (M // BLOCK_M) * (N // BLOCK_N),)
    per_block_quant_kernel[grid](
        X_ptr=x, SCALE_ptr=scale, OUT_ptr=x_int8,
        B=B, M=M, N=N,
        stride_b=x.stride(0), stride_m=x.stride(1), stride_n=x.stride(2),
        stride_sb=scale.stride(0), stride_sm=scale.stride(1), stride_sn=scale.stride(2),
        stride_ob=x_int8.stride(0), stride_om=x_int8.stride(1), stride_on=x_int8.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2
    )
    return x_int8, scale


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def int8_gemm_pipeline(A_bf, B_bf, out_dtype=torch.bfloat16):
    torch.cuda.set_device(A_bf.device)
    
    torch.cuda.synchronize()
    start = time.time()
    A_int8, A_scale = triton_per_block_quant(A_bf)
    B_int8, B_scale = triton_per_block_quant(B_bf)
    torch.cuda.synchronize()
    end = time.time()
    print(f'int8 time:{(end - start) * 1000} ms')

    torch.cuda.synchronize()
    start = time.time()
    C = triton_gemm(
        A_int8=A_int8,
        B_int8=B_int8,
        A_scale=A_scale,
        B_scale=B_scale,
        out_dtype=out_dtype
    )
    torch.cuda.synchronize()
    end = time.time()
    print(f'kernel time:{(end - start) * 1000} ms')
    
    return C


def bwd_33(a: torch.Tensor, b: torch.Tensor):
    C = int8_gemm_pipeline(a, b, out_dtype=torch.bfloat16)
    return C


if __name__ == "__main__":
    device = torch.device("cuda:2")
    B = 4
    M = 8192
    N = 8192
    K = 8192

    A_bf = torch.randn(B, M, K, device=device, dtype=torch.bfloat16)
    B_bf = torch.randn(B, N, K, device=device, dtype=torch.bfloat16)
    b_T = B_bf.transpose(1, 2)
    ref_c = torch.matmul(A_bf, b_T)
    

    C = int8_gemm_pipeline(A_bf, B_bf, out_dtype=torch.bfloat16)
    diff = calc_diff(C, ref_c)
    print(f"C: {C}")
    print(f"ref_c: {ref_c}")
    print(f"diff: {diff}")
    assert diff < 1e-3