import time
import torch


def time_block(label, fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = (time.time() - start) * 1000
    print(f"[TSS] {label} → {elapsed:.2f} ms")
    return result


if __name__ == "__main__":
    def matmul():
        a = torch.rand((2000, 2000), device='cpu')
        b = torch.rand((2000, 2000), device='cpu')
        return a @ b

    time_block("CPU matmul", matmul)

    if torch.cuda.is_available():
        def matmul_cuda():
            a = torch.rand((2000, 2000), device='cuda')
            b = torch.rand((2000, 2000), device='cuda')
            return a @ b
        time_block("CUDA matmul", matmul_cuda)
