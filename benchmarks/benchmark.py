"""
We focus on throughput here.

We will measure performance of our model, MFS and RPY
"""

import torch
import torch.nn as nn
import time

torch.set_float32_matmul_precision('high')

# import torch_tensorrt
# TENSORRT_AVAILABLE = True

device = torch.device("cuda")

def L1(d):
    """ Computes the outer product of each 3D vector in the batch with itself. """
    # d: [batch_size, 3]
    return torch.einsum('bi,bj->bij', d, d)  # [batch_size, 3, 3]

def L2(d):
    """ Returns the matrix (I - outer(d, d)) for each vector in the batch. """
    # Identity tensor expanded to batch size
    batch_size = d.shape[0]
    I = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)  # [batch_size, 3, 3]
    ddT = torch.einsum('bi,bj->bij', d, d)  # [batch_size, 3, 3]
    return I - ddT


# --------------------------------------------------
# Given hyperparameters
# --------------------------------------------------
viscosity = 1.0
eigens = None  # Not used in snippet, but left in for consistency

# --------------------------------------------------
# Define the ScNetwork
# --------------------------------------------------
class ScNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ScNetwork, self).__init__()
        lc = torch.zeros(3, 3, 3, dtype=torch.float)
        lc[0, 1, 2] = 1
        lc[1, 2, 0] = 1
        lc[2, 0, 1] = 1
        lc[0, 2, 1] = -1
        lc[2, 1, 0] = -1
        lc[1, 0, 2] = -1
        self.register_buffer("levi_civita", lc)
        self.inv_visc = 1.0/viscosity

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
        )

    def L3(self, d):
        """
        Computes the cross-product matrix for each 3D vector in the batch.
        Because we registered self.levi_civita as a buffer, we can reference it here.
        """
        # Using einsum for batched matrix-vector multiplication:
        return torch.einsum('ijk,bk->bij', self.levi_civita, d)

    def forward(self, X, force):
        d_vec, r = X[:, :3], X[:, 3]

        sc = self.layers(X)

        # build the mobility matrix
        d_vec = d_vec / r.unsqueeze(-1)
        RT = sc[:, 2].unsqueeze(1).unsqueeze(2) * self.L3(d_vec)

        K = torch.empty((len(X), 6, 6), dtype=torch.float32, device=X.device)
        K[:, :3, :3] = sc[:, 0].unsqueeze(1).unsqueeze(2) * L1(d_vec) \
             + sc[:, 1].unsqueeze(1).unsqueeze(2) * L2(d_vec)
        K[:, 3:, :3] = RT
        K[:, :3, 3:] = RT.transpose(1,2)
        K[:, 3:, 3:] = sc[:, 3].unsqueeze(1).unsqueeze(2) * L1(d_vec) \
             + sc[:, 4].unsqueeze(1).unsqueeze(2) * L2(d_vec)
    

        # Finally compute velocity
        M = K * self.inv_visc
        velocity = torch.bmm(M, force.unsqueeze(-1)).squeeze(-1)
        return velocity
    


# --------------------------------------------------
# Throughput measurement utility
# --------------------------------------------------
import nvtx

input_dim = 4


def measure_throughput(model, device, batch_size, n_warmup=1, n_iter=3):
    """Measures throughput (samples/sec) on random data for a given model and batch size."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    X = torch.randn(batch_size, input_dim, device=device)
    force = torch.randn(batch_size, 6, device=device)

    # Warm-up
    with nvtx.annotate("warmup"):
        with torch.inference_mode():
            for _ in range(n_warmup):
                _ = model(X, force)
    torch.cuda.synchronize(device)

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        with torch.inference_mode():
            _ = model(X, force)
    end.record()
    torch.cuda.synchronize(device)
    elapsed_sec = start.elapsed_time(end) / 1000.0

    # Return throughput = samples / total_time
    total_samples = batch_size * n_iter
    return (total_samples / elapsed_sec) / 1e6   # M samples/sec

# --------------------------------------------------
# Main script
# --------------------------------------------------
def bench():
    # Create the base model and move to device
    base_model = ScNetwork(input_dim).to(device)

    # --------------------------------------------------
    # 1) Baseline PyTorch model (without compile)
    # --------------------------------------------------
    print("\n-- Baseline PyTorch --")
    batch_candidates = [1024, 8192, 16384, 24064, 32768, 65536, ]
                        #2**17, 2**18, 2**19, 2**20, 2**21, 2**22, 
                        #2**23, 2**24, 2**25]
    #batch_candidates = [16384]
    best_throughput_pt = 0
    best_batch_pt = 1

    for bs in batch_candidates:
        thpt = measure_throughput(base_model, device, bs)
        print(f"Batch size={bs}, throughput={thpt:.2f} Msamples/sec")
        if thpt > best_throughput_pt:
            best_throughput_pt = thpt
            best_batch_pt = bs

    print(f"Best batch size (PyTorch)={best_batch_pt}, best throughput={best_throughput_pt:.2f}\n")

    # --------------------------------------------------
    # 2) PyTorch 2.0+ compile
    # --------------------------------------------------
    print("-- torch.compile --")
    if hasattr(torch, "compile"):
        compiled_model = torch.compile(base_model,fullgraph=True)
        best_throughput_compile = 0
        best_batch_compile = 1
        
        for bs in batch_candidates:
            thpt = measure_throughput(compiled_model, device, bs)
            print(f"Batch size={bs}, throughput={thpt:.2f} Msamples/sec")
            if thpt > best_throughput_compile:
                best_throughput_compile = thpt
                best_batch_compile = bs
        
        print(f"Best batch size (torch.compile)={best_batch_compile}, "
              f"best throughput={best_throughput_compile:.2f}\n")
    else:
        print("torch.compile not available in this PyTorch version.\n")

    # --------------------------------------------------
    # 3) TensorRT via torch-tensorrt
    # --------------------------------------------------
    # print("-- TensorRT (torch_tensorrt) --")
    # if TENSORRT_AVAILABLE:
    #     trt_best_throughput = 0
    #     trt_best_batch = 1
        
    #     for bs in batch_candidates:
    #         # We compile for a specific batch size dimension
    #         # If your real workload has variable shapes, you can
    #         # create a range of min/opt/max shapes or use dynamic shape.
    #         trt_inputs = [
    #             torch_tensorrt.Input((bs, input_dim), dtype=torch.float32),
    #             torch_tensorrt.Input((bs, 6), dtype=torch.float32),
    #         ]
                        
    #         trt_model = torch_tensorrt.compile(
    #             base_model,
    #             inputs=trt_inputs,
    #             enabled_precisions={torch.float32},  
    #             truncate_long_and_double=True
    #         )
    #         thpt = measure_throughput(trt_model, device, bs)
    #         print(f"Batch size={bs}, throughput={thpt:.2f} samples/sec")
    #         if thpt > trt_best_throughput:
    #             trt_best_throughput = thpt
    #             trt_best_batch = bs

    #     print(f"Best batch size (TensorRT)={trt_best_batch}, "
    #           f"best throughput={trt_best_throughput:.2f} M samples/sec\n")
    # else:
    #     print("TensorRT is not available in this environment.\n")

if __name__ == "__main__":
    bench()

    if torch.cuda.is_available():
        max_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\nTotal Peak VRAM Usage: {max_vram_gb:.2f} GB")