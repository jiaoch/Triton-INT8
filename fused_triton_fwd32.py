# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# Copyright (C) Zhejiang Lab(ZJLab) 2025. All Right Reserved.
# FilePath: \undefinedc:\Users\hp\Desktop\fused_triton_fwd32.py
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

def ceildiv(a, b):
    return (a + b - 1) // b


# @triton.autotune(
#   configs=[
#     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
#     triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
#     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
#     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=3, num_warps=8),
#     triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
#     triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=3, num_warps=8),
#     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 256}, num_stages=3, num_warps=8),
#     triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256}, num_stages=3, num_warps=8),
#   ],
#   key=['M','N','K']
# )
@triton.jit
def fused_int8_gemm_kernel(
    A_bf_ptr, B_bf_ptr, C_ptr,
    K,
    stride_ab, stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a_bf_ptrs = A_bf_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a_bf = tl.load(a_bf_ptrs)

        b_bf_ptrs = B_bf_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        b_bf = tl.load(b_bf_ptrs)

        amax_a = tl.max(tl.abs(a_bf)) + 1e-5
        scale_a = 127.0 / amax_a
        a_q = tl.floor(a_bf * scale_a + 0.5).to(tl.int8)
        # a_qf = tl.floor(a_bf * scale_a + 0.5)         
        # a_qf = tl.where(a_qf > 127.0, 127.0, a_qf)
        # a_qf = tl.where(a_qf < -128.0, -128.0, a_qf)
        # a_q = tl.cast(a_qf, tl.int8) 
        
        amax_b = tl.max(tl.abs(b_bf), axis=0) + 1e-5
        scale_b = 127.0 / amax_b
        b_q = tl.floor(b_bf * scale_b[None, :] + 0.5).to(tl.int8)
        # b_qf = tl.floor(b_bf * scale_b[None, :] + 0.5) 
        # b_qf = tl.where(b_qf > 127.0, 127.0, b_qf)
        # b_qf = tl.where(b_qf < -128.0, -128.0, b_qf)
        # b_q = tl.cast(b_qf, tl.int8) 
 
        acc += tl.dot(a_q, b_q) / (scale_a * scale_b[None, :])
        
    
    c_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)

def fused_int8_gemm(A_bf, B_bf):
    B_dim, M, K = A_bf.shape
    N = B_bf.shape[0]
    A_bf = A_bf.contiguous()
    #B_bf = B_bf.t().contiguous()
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    C = torch.empty((B_dim, M, N), device=A_bf.device, dtype=torch.bfloat16)
    
    grid = (
        B_dim,
        ceildiv(M, BLOCK_M),
        ceildiv(N, BLOCK_N),
    )

    fused_int8_gemm_kernel[grid](
        A_bf_ptr=A_bf, B_bf_ptr=B_bf, C_ptr=C,
        K=K,
        stride_ab=A_bf.stride(0), stride_am=A_bf.stride(1), stride_ak=A_bf.stride(2),
        stride_bn=B_bf.stride(0), stride_bk=B_bf.stride(1),
        stride_cb=C.stride(0), stride_cm=C.stride(1), stride_cn=C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3
    )
    return C

def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def fwd_32(a: torch.Tensor, b: torch.Tensor):
    a = a.cuda()
    b = b.cuda()
    C = fused_int8_gemm(A_bf=a, B_bf=b)
    return C

if __name__ == "__main__":
    device = torch.device("cuda:0")
    B = 4
    M = 8192
    N = 8192
    K = 8192

    A_bf = torch.randn(B, M, K, device=device, dtype=torch.bfloat16)
    B_bf = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    ref_c = torch.matmul(A_bf, B_bf.T)
    
    C = fused_int8_gemm(A_bf, B_bf)
    diff = calc_diff(C, ref_c)
    print(f"diff: {diff}")
    assert diff < 1e-3