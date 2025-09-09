import torch
import triton
import triton.language as tl
from typing import Tuple
import time


def ceildiv(a, b):
    return (a + b - 1) // b


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
    stride_a_scale_b, stride_a_scale_m, stride_a_scale_k,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    b_scale_ptr = B_scale_ptr + offs_n * stride_b_scale_n 
    b_scale_vec = tl.load(b_scale_ptr)

    for k_iter in range(0, K, BLOCK_K):
        k_block_idx = k_iter // BLOCK_K
        a_ptrs = a_ptr_base + (k_iter + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr_base + (k_iter + offs_k[:, None]) * stride_bk
        a_q = tl.load(a_ptrs)
        b_q = tl.load(b_ptrs)

        a_scale_val = tl.load(A_scale_ptr + pid_b * stride_a_scale_b + pid_m * stride_a_scale_m + k_block_idx * stride_a_scale_k)

        # b_scale_ptr = B_scale_ptr + k_block_idx * stride_b_scale_k + offs_n * stride_b_scale_n 
        # b_scale_vec = tl.load(b_scale_ptr)
        
        acc += tl.dot(a_q, b_q) * a_scale_val * b_scale_vec[None, :]

    c_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)

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
        stride_a_scale_b=A_scale.stride(0), stride_a_scale_m=A_scale.stride(1), stride_a_scale_k=A_scale.stride(2),
        stride_b_scale_n=B_scale.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        # num_warps=8, num_stages=3
    )
    return C




@triton.jit
def per_token_quant_kernel(
    X_ptr, SCALE_ptr, OUT_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_sm, stride_sn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1) 
    offs_m = tl.multiple_of(pid_m * BLOCK_M + tl.arange(0, BLOCK_M), BLOCK_M)
    offs_n = tl.multiple_of(pid_n * BLOCK_N + tl.arange(0, BLOCK_N), BLOCK_N)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.where(amax > 1e-4, amax, 1e-4)
    scale = 127.0 / amax
    x_scaled = x * scale[:, None]
    # x_q = tl.floor(x_scaled + 0.5).to(tl.int8)
    x_q = tl.minimum(tl.maximum(tl.floor(x_scaled + 0.5), -128), 127).to(tl.int8)

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, x_q)
    scale_inv = 1.0 / scale
    scale_ptrs = SCALE_ptr + offs_m * stride_sm + pid_n * stride_sn
    tl.store(scale_ptrs, scale_inv)


def triton_per_token_quant(x: torch.Tensor):
    assert x.dim() == 2 and x.shape[1] % 128 == 0
    M, N = x.shape
    BLOCK_M = 128
    BLOCK_N = N
    x = x.contiguous()
    device = x.device

    x_int8 = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((M, N // BLOCK_N), dtype=torch.float32, device=device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    per_token_quant_kernel[grid](
        X_ptr=x, SCALE_ptr=scale, OUT_ptr=x_int8,
        M=M, N=N,
        stride_xm=x.stride(0), stride_xn=x.stride(1),
        stride_sm=scale.stride(0), stride_sn=scale.stride(1),
        stride_om=x_int8.stride(0), stride_on=x_int8.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3
    )

    return x_int8, scale


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
    # x_q = tl.floor(x_scaled + 0.5).to(tl.int8)
    x_q = tl.minimum(tl.maximum(tl.floor(x_scaled + 0.5), -128), 127).to(tl.int8)

    out_ptrs = (OUT_ptr + b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, x_q)
    scale_inv = 1.0 / scale
    scale_ptr = (SCALE_ptr + b * stride_sb + m_block * stride_sm + n_block * stride_sn)
    tl.store(scale_ptr, scale_inv)

def triton_per_block_quant(x: torch.Tensor):
    B, M, N = x.shape
    BLOCK_M = 128
    BLOCK_N = 128
    x = x.contiguous()
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
        num_warps=8, num_stages=3
    )
    return x_int8, scale



def int8_gemm_pipeline(A_bf, B_bf, out_dtype=torch.bfloat16):
    
    torch.cuda.set_device(A_bf.device)
    torch.cuda.synchronize()
    start = time.time()
    A_int8, A_scale = triton_per_block_quant(A_bf)
    B_int8, B_scale = triton_per_token_quant(B_bf)
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




def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim



def fwd_32(a: torch.Tensor, b: torch.Tensor):
    C = int8_gemm_pipeline(a, b, out_dtype=torch.bfloat16)
    return C


if __name__ == "__main__":
    device = torch.device("cuda:5")
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

