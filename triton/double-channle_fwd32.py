import torch
import triton
import triton.language as tl
from typing import Tuple
import time
import math
from math import ceil
from fast_hadamard_transform import hadamard_transform


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],   
)
@triton.jit
def int8_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    A_scale_ptr, B_scale_ptr,
    B, M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_ab, stride_cb,
    stride_a_scale_b, stride_a_scale_m,
    stride_b_scale_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_m = tl.multiple_of(pid_m * BLOCK_M + tl.arange(0, BLOCK_M), BLOCK_M)
    offs_n = tl.multiple_of(pid_n * BLOCK_N + tl.arange(0, BLOCK_N), BLOCK_N)
    offs_k = tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K)

    a_ptr_base = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am
    b_ptr_base = B_ptr + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    b_scale_ptr = B_scale_ptr + offs_n * stride_b_scale_n 
    b_scale_vec = tl.load(b_scale_ptr)
    a_scale_ptr = A_scale_ptr + pid_b * stride_a_scale_b + offs_m * stride_a_scale_m 
    a_scale_vac = tl.load(a_scale_ptr)

    for k_iter in range(0, K, BLOCK_K):
        a_ptrs = a_ptr_base + (k_iter + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr_base + (k_iter + offs_k[:, None]) * stride_bk
        a_q = tl.load(a_ptrs)
        b_q = tl.load(b_ptrs)
        acc += tl.dot(a_q, b_q) 

    c_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc * a_scale_vac[:, None] * b_scale_vec[None, :])


def triton_gemm(A_int8, B_int8, A_scale, B_scale, out_dtype=torch.bfloat16):
    B, M, K = A_int8.shape
    N = B_int8.shape[0]
    A_int8 = A_int8.contiguous()
    B_int8 = B_int8.contiguous()
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    C = torch.zeros((B, M, N), device=A_int8.device, dtype=out_dtype)
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        B,
    )

    int8_gemm_kernel[grid](
        A_ptr=A_int8, B_ptr=B_int8, C_ptr=C,
        A_scale_ptr=A_scale, B_scale_ptr=B_scale,
        B=B, M=M, N=N, K=K,
        stride_am=A_int8.stride(1), stride_ak=A_int8.stride(2),
        stride_bn=B_int8.stride(0), stride_bk=B_int8.stride(1),
        stride_cm=C.stride(1), stride_cn=C.stride(2),
        stride_ab=A_int8.stride(0), stride_cb=C.stride(0),
        stride_a_scale_b=A_scale.stride(0), stride_a_scale_m=A_scale.stride(1),
        stride_b_scale_n=B_scale.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        # num_warps=8, num_stages=3
    )
    return C


@triton.jit
def _pass1_tile_2d(
    X_ptr, PARTIAL_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_pm, stride_pn,
    BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)
    n_block = tl.program_id(1)
    if row >= M:
        return

    cols = n_block * BLOCK_N + tl.arange(0, BLOCK_N) 
    ptrs = X_ptr + row * stride_xm + cols * stride_xn
    vals = tl.load(ptrs)  
    abs_vals = tl.abs(vals)
    tile_max = tl.max(abs_vals, axis=0)  
    tl.store(PARTIAL_ptr + row * stride_pm + n_block * stride_pn, tile_max)


@triton.jit
def _pass2_tile_2d(
    X_ptr, SCALE_mul_ptr, OUT_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_sm,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)
    n_block = tl.program_id(1)
    if row >= M:
        return
    cols = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    in_ptrs = X_ptr + row * stride_xm + cols * stride_xn
    vals = tl.load(in_ptrs) 
    scale_mul = tl.load(SCALE_mul_ptr + row * stride_sm) 
    x_scaled = vals * scale_mul
    x_rounded = tl.floor(x_scaled + 0.5)
    x_clamped = tl.minimum(tl.maximum(x_rounded, -128.0), 127.0)
    x_q = x_clamped.to(tl.int8)
    out_ptrs = OUT_ptr + row * stride_om + cols * stride_on
    tl.store(out_ptrs, x_q)


def triton_per_row_quant_2d(x: torch.Tensor, BLOCK_N: int = 2048):
    x = x.contiguous()
    M, N = x.shape
    device = x.device
    BLOCK_N = int(BLOCK_N)
    num_n = ceil(N / BLOCK_N)
    partial = torch.empty((M, num_n), dtype=torch.float32, device=device)
    grid1 = (M, num_n)
    _pass1_tile_2d[grid1](
        x, partial,
        M, N,
        x.stride(0), x.stride(1),
        partial.stride(0), partial.stride(1),
        BLOCK_N=BLOCK_N,
        # num_warps=num_warps, num_stages=num_stages
    )

    amax = torch.max(partial, dim=1).values 
    amax = torch.clamp(amax, min=1e-4) 
    scale_mul = (127.0 / amax) 
    scale_inv = (amax / 127.0) 
    x_int8 = torch.empty_like(x, dtype=torch.int8)
    scale_mul = scale_mul.contiguous()

    grid2 = (M, num_n)
    _pass2_tile_2d[grid2](
        x, scale_mul, x_int8,
        M, N,
        x.stride(0), x.stride(1),
        scale_mul.stride(0),
        x_int8.stride(0), x_int8.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2
    )

    return x_int8, scale_inv


@triton.jit
def _pass1_tile_3d(
    X_ptr, PARTIAL_ptr,
    B, M, N,
    stride_xb, stride_xm, stride_xn,
    stride_pb, stride_pm, stride_pn,
    BLOCK_N: tl.constexpr
):

    b = tl.program_id(0)
    row = tl.program_id(1)
    n_block = tl.program_id(2)
    if (b >= B) or (row >= M):
        return
    cols = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    ptrs = X_ptr + b * stride_xb + row * stride_xm + cols * stride_xn
    vals = tl.load(ptrs)
    abs_vals = tl.abs(vals)
    tile_max = tl.max(abs_vals, axis=0)
    tl.store(PARTIAL_ptr + b * stride_pb + row * stride_pm + n_block * stride_pn, tile_max)


@triton.jit
def _pass2_tile_3d(
    X_ptr, SCALE_mul_ptr, OUT_ptr,
    B, M, N,
    stride_xb, stride_xm, stride_xn,
    stride_sm, stride_srow,
    stride_ob, stride_om, stride_on,
    BLOCK_N: tl.constexpr
):

    b = tl.program_id(0)
    row = tl.program_id(1)
    n_block = tl.program_id(2)
    if (b >= B) or (row >= M):
        return

    cols = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    in_ptrs = X_ptr + b * stride_xb + row * stride_xm + cols * stride_xn
    vals = tl.load(in_ptrs)
    scale_mul = tl.load(SCALE_mul_ptr + b * stride_sm + row * stride_srow) 
    x_scaled = vals * scale_mul
    x_rounded = tl.floor(x_scaled + 0.5)
    x_clamped = tl.minimum(tl.maximum(x_rounded, -128.0), 127.0)
    x_q = x_clamped.to(tl.int8)

    out_ptrs = OUT_ptr + b * stride_ob + row * stride_om + cols * stride_on
    tl.store(out_ptrs, x_q)


def triton_per_row_quant_3d(x: torch.Tensor, BLOCK_N: int = 2048):

    x = x.contiguous()
    B, M, N = x.shape
    device = x.device

    BLOCK_N = int(BLOCK_N)
    num_n = ceil(N / BLOCK_N)
    partial = torch.empty((B, M, num_n), dtype=torch.float32, device=device)

    grid1 = (B, M, num_n)
    _pass1_tile_3d[grid1](
        x, partial,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        partial.stride(0), partial.stride(1), partial.stride(2),
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2
    )

    amax = torch.max(partial, dim=2).values 
    amax = torch.clamp(amax, min=1e-4)
    scale_mul = (127.0 / amax)
    scale_inv = (amax / 127.0)
    x_int8 = torch.empty_like(x, dtype=torch.int8)

    grid2 = (B, M, num_n)
    _pass2_tile_3d[grid2](
        x, scale_mul, x_int8,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        scale_mul.stride(0), scale_mul.stride(1),
        x_int8.stride(0), x_int8.stride(1), x_int8.stride(2),
        BLOCK_N=BLOCK_N,
        # num_warps=num_warps, num_stages=num_stages
    )

    return x_int8, scale_inv


def int8_gemm_pipeline(A_bf, B_bf, out_dtype=torch.bfloat16):
    
    inv_K = 1.0 / A_bf.shape[-1]
    torch.cuda.set_device(A_bf.device)
    A_bf = hadamard_transform(A_bf) 
    B_bf = hadamard_transform(B_bf) 
    A_int8, A_scale = triton_per_row_quant_3d(A_bf)
    B_int8, B_scale = triton_per_row_quant_2d(B_bf)
    C = triton_gemm(
        A_int8=A_int8,
        B_int8=B_int8,
        A_scale=A_scale,
        B_scale=B_scale,
        out_dtype=out_dtype
    )
    C = C * inv_K
    return C


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def fwd_32(a: torch.Tensor, b: torch.Tensor):
    torch.cuda.set_device(a.device)
    
    inv_K = 1.0 / a.shape[-1]
    a = hadamard_transform(a) 
    b = hadamard_transform(b) 
    A_int8, A_scale = triton_per_row_quant_3d(a)
    B_int8, B_scale = triton_per_row_quant_2d(b)
    C = triton_gemm(
        A_int8=A_int8,
        B_int8=B_int8,
        A_scale=A_scale,
        B_scale=B_scale,
        out_dtype=torch.bfloat16
    )
    C = C * inv_K
    return C


if __name__ == "__main__":
    device = torch.device("cuda:6")
    B = 4
    M = 8192
    N = 8192
    K = 8192

    A_bf = torch.randn(B, M, K, device=device, dtype=torch.bfloat16)
    B_bf = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    ref_c = torch.matmul(A_bf, B_bf.T)
    C = int8_gemm_pipeline(A_bf, B_bf, out_dtype=torch.bfloat16)
    diff = calc_diff(C, ref_c)
    print(f"C: {C}")
    print(f"ref_c: {ref_c}")
    print(f"diff: {diff}")
    assert diff < 1e-3
