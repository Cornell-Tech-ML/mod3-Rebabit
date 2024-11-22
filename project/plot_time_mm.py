import time
import numpy as np
import matplotlib.pyplot as plt
import minitorch

# Define backends
CPUBackend = minitorch.TensorBackend(minitorch.FastOps)
CudaBackend = minitorch.TensorBackend(minitorch.CudaOps)

# Function to run matrix multiplication
def execute_matmul(backend_type, matrix_dim):
    batch_count = 2
    matrix_a = minitorch.rand((batch_count, matrix_dim, matrix_dim), backend=backend_type)
    matrix_b = minitorch.rand((batch_count, matrix_dim, matrix_dim), backend=backend_type)
    result = matrix_a @ matrix_b

# Benchmarking function
def evaluate_performance():
    dimensions = [32, 64, 128, 256, 512, 1024]
    num_trials = 5
    timings = {"CPU": [], "GPU": []}

    # Warm-up phase for stabilization
    execute_matmul(CPUBackend, 64)
    execute_matmul(CudaBackend, 64)

    for dim in dimensions:
        cpu_durations, gpu_durations = [], []

        for _ in range(num_trials):
            # Measure time for CPUBackend
            start_time = time.time()
            execute_matmul(CPUBackend, dim)
            cpu_durations.append(time.time() - start_time)

            # Measure time for CudaBackend
            start_time = time.time()
            execute_matmul(CudaBackend, dim)
            gpu_durations.append(time.time() - start_time)

        # Store average time for each backend
        timings["CPU"].append(np.mean(cpu_durations))
        timings["GPU"].append(np.mean(gpu_durations))
        print(dim, ":", " CPU: ",timings["CPU"][-1], "GPU: ", timings["GPU"][-1])

    return dimensions, timings

# Plotting function
def visualize_results(dimensions, timings):
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, timings["CPU"], label="CPUBackend", marker="^")
    plt.plot(dimensions, timings["GPU"], label="CudaBackend", marker="D")
    plt.xlabel("Matrix Dimension")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Performance Comparison of Matrix Multiplication")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run benchmarking and plot results
matrix_sizes, runtime_results = evaluate_performance()
visualize_results(matrix_sizes, runtime_results)