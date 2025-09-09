import torch
import time
from double_channel_triton_fwd import fwd_32
from double_channel_triton_bwd import bwd_33


def bf16_training(a, b):
    b_T = b.transpose(-2, -1)
    C = torch.matmul(a, b_T)
    return C


def measure_latency(func, *args, total_repeat=80, warmup=20, device=None):
    times = []
    for i in range(total_repeat):
        if i < warmup:
            func(*args)   
            continue
        torch.cuda.synchronize(device)
        start = time.time()
        func(*args)
        torch.cuda.synchronize(device)
        end = time.time()
        times.append(end - start)
    avg_time_ms = sum(times) / len(times) * 1000
    return avg_time_ms


if __name__ == "__main__":
    device = torch.device("cuda:2")
    
    a = torch.randn(4, 8192, 8192, device=device, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    bf16_time = measure_latency(bf16_training, a, b, device = device)
    quant_time = measure_latency(fwd_32, a, b, device = device)
    print(f"fwd32_bf16 time: {bf16_time:.2f} ms")
    print(f"fwd32_int8 time: {quant_time:.2f} ms")
    print(f"fwd32_speedup: {100 * abs(bf16_time - quant_time) / bf16_time:.2f}%")
    
    print("--------------------------------------------------")
    
    a = torch.randn(4, 8192, 8192, device=device, dtype=torch.bfloat16)
    b = torch.randn(4, 8192, 8192, device=device, dtype=torch.bfloat16)
    bf16_time = measure_latency(bf16_training, a, b, device = device)
    quant_time = measure_latency(bwd_33, a, b, device = device)
    print(f"bwd33_bf16 time: {bf16_time:.2f} ms")
    print(f"bwd33_int8 time: {quant_time:.2f} ms")
    print(f"bwd33_speedup: {100 * abs(bf16_time - quant_time) / bf16_time:.2f}%")

    
    







    













    







