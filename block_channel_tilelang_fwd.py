from typing import Tuple

import triton
import triton.language as tl
import torch
import tilelang.testing
import tilelang
import tilelang.language as T
from tilelang.utils.tensor import map_torch_type
import time

tilelang.testing.set_random_seed(42)


@tilelang.jit
def tl_gemm(
    B,
    M,
    N,
    K,
    block_N,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    assert in_dtype in [
        "int8",
    ], "Currently only int8 is supported"
    assert out_dtype in [
        "bfloat16"
    ], "Currently only bfloat16 is supported"

    group_size = 256
    block_M = 128
    block_K = 128
    batch_zize = B

    A_shape = (B, M, K)
    Scales_A_shape = (B, T.ceildiv(M, block_M), T.ceildiv(K, group_size))
    B_shape = (N, K)
    Scales_B_shape = (N,)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (block_M, block_N)

    @T.prim_func
    def main(
            a: T.Tensor(A_shape, in_dtype),
            b: T.Tensor(B_shape, in_dtype),
            c: T.Tensor((B, M, N), out_dtype),
            scales_a: T.Tensor(Scales_A_shape, "float32"),
            scales_b: T.Tensor(Scales_B_shape, "float32"),
    ):
        with T.Kernel(batch_zize, T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bz, bx, by):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype)
            Scale_C_shared = T.alloc_shared((block_N), "float32")
            C_local = T.alloc_fragment(C_shared_shape, accum_dtype)
            C_local_accum = T.alloc_fragment(C_shared_shape, "float32")
            
            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                # Load A into shared memory
                T.copy(a[bz, by * block_M, k * block_K], A_shared)
                # Load B into shared memory
                T.copy(b[bx * block_N, k * block_K], B_shared)
                # Load scale into shared memory
                for i in T.Parallel(block_N):
                    Scale_C_shared[i] = scales_a[bz, by, (k * block_K) // group_size] * scales_b[bx * block_N + i]

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Promote to enable 2xAcc
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[j]

                T.clear(C_local)
            # TMA store
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, c[bz, by * block_M, bx * block_N])

    return main


def ceildiv(a, b):
    return (a + b - 1) // b

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
    x_q = tl.floor(x_scaled + 0.5).to(tl.int8)

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
    scale = torch.empty((M, N // BLOCK_N), dtype=torch.bfloat16, device=device)

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

    return x_int8, scale.float()


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
    BLOCK_N = 256
    x = x.contiguous()
    x_int8 = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((B, M // BLOCK_M, N // BLOCK_N), dtype=torch.bfloat16, device=x.device)

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
    return x_int8, scale.float()


# def per_token_cast_to_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     m, n = x.shape
#     x_view = x.view(m, -1, 128)
#     x_amax = x_view.abs().amax(dim=2, keepdim=True).clamp(min=1e-4)
#     scale = 127.0 / x_amax
#     x_int8 = (x_view * scale).round().clamp(-127, 127).to(torch.int8)
    
#     return x_int8.view(m, n), (1.0 / scale).view(m, -1)  

def per_token_cast_to_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    x_amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)
    scale = 127.0 / x_amax
    x_int8 = (x * scale).round().clamp(-127, 127).to(torch.int8)
    return x_int8.view(m, n), (1.0 / scale).view(m, -1) 

def per_block_cast_to_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    b, m, n = x.shape
    x_view = x.view(b, m // 128, 128, n // 128, 128)
    x_amax = x_view.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-4)
    scale = 127.0 / x_amax
    x_int8 = (x_view * scale).round().clamp(-127, 127).to(torch.int8)
    x_int8 = x_int8.view(b, m, n)
    return x_int8, (1.0 / scale).view(b, m // 128, n // 128)




def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def assert_tl_gemm_correctness(B, M, N, K, block_N, in_dtype, out_dtype, accum_dtype):
    kernel = tl_gemm(B, M, N, K, block_N, in_dtype, out_dtype, accum_dtype)
    src_code = kernel.get_kernel_source()

    # src_code is the generated cuda source
    assert src_code is not None

    in_dtype = map_torch_type(in_dtype)
    out_dtype = map_torch_type(out_dtype)
    accum_dtype = map_torch_type(accum_dtype)

    a = torch.randn(B, M, K).to(torch.bfloat16).cuda()
    b = torch.randn(N, K).to(torch.bfloat16).cuda()
    # A_int8, A_scale = per_block_cast_to_int8(a)
    # B_int8, B_scale = per_token_cast_to_int8(b)
    A_int8, A_scale = triton_per_block_quant(a)
    B_int8, B_scale = triton_per_token_quant(b)

    C = torch.zeros(B, M, N, device="cuda", dtype=out_dtype)

    kernel(A_int8, B_int8, C, A_scale, B_scale)
    ref_c = torch.matmul(a, b.T)
    diff = calc_diff(C, ref_c)
    print(f"C: {C}")
    print(f"ref_c: {ref_c}")
    print(f"diff: {diff}")
    assert diff < 1e-3


def fwd_32(a, b):
    torch.cuda.set_device(a.device)
    B, M, N, K = a.shape[0], a.shape[1], b.shape[0], a.shape[2]
    block_N = 128
    torch.cuda.synchronize()
    start = time.time()
    # A_int8, A_scale = per_block_cast_to_int8(a)
    # B_int8, B_scale = per_token_cast_to_int8(b)
    A_int8, A_scale = triton_per_block_quant(a)
    B_int8, B_scale = triton_per_token_quant(b)
    torch.cuda.synchronize()
    end = time.time()
    print(f'int8 time:{(end - start) * 1000} ms')
    C = torch.zeros(B, M, N, device=a.device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = time.time()
    kernel = tl_gemm(B, M, N, K, block_N, "int8", "bfloat16", "int32")
    kernel(A_int8, B_int8, C, A_scale, B_scale)
    torch.cuda.synchronize()
    end = time.time()
    print(f'kernel time:{(end - start) * 1000} ms')
    return C



if __name__ == "__main__":
    for dtype in ["int8"]:
        for out_dtype in ["bfloat16"]:
            for block_N in [128]:
                assert_tl_gemm_correctness(4, 1024, 1024, 8192, block_N, dtype, out_dtype, "int32")






