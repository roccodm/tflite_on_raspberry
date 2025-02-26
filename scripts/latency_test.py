# Released under CC0 License. Author: Rocco De Marco - CNR IRBIM Ancona  
#  
# This script benchmarks the inference latency of a TFLite model across different  
# thread counts. It provides statistics such as average, minimum, and maximum latency.  
#  
# Usage: python3 latency_test.py <model_path> <num_runs>  
#   - model_path: Path to the TFLite model (e.g., "models/optimized_model_1.tflite").  
#   - num_runs: Number of inference executions to perform for each thread count.  
#  
# Output: Latency statistics (in milliseconds) for each thread count, printed to the console.  

import time
import numpy as np
import tflite_runtime.interpreter as tflite
import json
import os
import sys

WARM_UP_RUNS = 5  # warn-up executions
DEBUG = True

def run_latency_test(model_path, num_runs):
    for num_thread in range (1,9):
        try:
            interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_thread)
            interpreter.allocate_tensors()
        except Exception as e:
            print(f"Error while loading model {model_path}: {e}")
            continue
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        input_data = np.random.rand(*input_shape).astype(np.float32)
        latencies = []
        # warm-up
        if DEBUG:
            print(f"[DEBUG] Starting warm-up for model {model_path}...")
        for _ in range(WARM_UP_RUNS):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        # Execute latency test for num_runs times
        for run in range(num_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            latencies.append(latency)
            if DEBUG and run < 5:  # Print log for first five tests
                print(f"[DEBUG] Executions {run + 1}: {latency:.2f} ms")
        # Statistics
        avg_latency = sum(latencies) / num_runs
        min_latency = min(latencies)
        max_latency = max(latencies)
        sd = np.std(latencies)
        # Prepare report
        results = {
                "model": model_path,
                "num_runs": num_runs,
                "num_threads": num_thread,
                "average_latency_ms": round(avg_latency, 4),
                "sd": round(sd,4),
                "min_latency_ms": round(min_latency, 4),
                "max_latency_ms": round(max_latency, 4),
                "latencies": latencies  # Salva tutti i valori
            }
        print(results)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <models_path> <num_runs>")
        sys.exit(1)
    model_path, num_runs = sys.argv[1], int(sys.argv[2])
    run_latency_test(model_path, num_runs)
