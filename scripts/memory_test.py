# Released under CC0 License. Author: Rocco De Marco - CNR IRBIM Ancona  
#  
# This script measures memory usage during different phases of TFLite model execution:  
# 1. Library initialization  
# 2. Random image generation  
# 3. Model loading  
# 4. Tensor allocation  
# 5. Inference execution  
#  
# Usage: python3 memory_test.py <model_path> <num_threads> <num_images>  
#   - model_path: Path to the TFLite model (e.g., "models/optimized_model_1.tflite").  
#   - num_threads: Number of threads for the TFLite interpreter (1-8).  
#   - num_images: Number of random images to generate and process.  
#  
# Output: Memory usage (in MB) for each phase, printed to the console.  

import time
import numpy as np
import tflite_runtime.interpreter as tflite
import psutil
import sys
import os

# Set DEBUG=True for more verbose output
DEBUG = True

def log(msg):
    if DEBUG:
        print(msg)

def memory_usage_mb():
    process = psutil.Process()
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)

def worker(model_path, num_images, num_threads):
    try:

        # 1. libraries loaded
        time.sleep(0.5)
        mem_state0 = memory_usage_mb()
        log(f"Memory after process start: {mem_state0:.1f}")

        # 2. random test images allocation
        input_shape = (1, 300, 150, 1)
        images = np.random.rand(num_images, *input_shape[1:]).astype(np.float32)
        mem_state1 = memory_usage_mb()
        log(f"Memory after generating {num_images} images: {mem_state1:.1f}")

        # 3. model load with buffer
        with open(model_path, 'rb') as f:
            model_content = f.read()

        # getting real size of the model
        model_size_mb = os.path.getsize(model_path)/(1024*1024)
        interpreter = tflite.Interpreter(
            model_content=model_content,
            num_threads=num_threads
        )
        time.sleep(0.5)
        mem_state2 = memory_usage_mb()
        log(f"Memory after model load ({model_size_mb:.1f} MB file): {mem_state2:.1f}")

        # 4. tensor alloc with warmup
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()

        # warmup
        interpreter.set_tensor(input_details[0]['index'], np.zeros(input_details[0]['shape'], dtype=np.float32))
        interpreter.invoke()
        mem_state3 = memory_usage_mb()
        log(f"Memory after tensor alloc: {mem_state3:.1f}")

        # 5. Inference
        for idx in range(num_images):
            interpreter.set_tensor(input_details[0]['index'], images[idx:idx+1])
            interpreter.invoke()

        mem_state4 = memory_usage_mb()
        log(f"Memory after inference: {mem_state4:.1f}")

        # Output
        log("model_path, num_threads, num_images, model_real_size, after_libraries, after_image_creation, after_model_load, after_tensor_alloc, after_inference")
        print(f"{model_path},{num_threads},{num_images},{model_size_mb:.1f},{mem_state0:.1f},{mem_state1:.1f},{mem_state2:.1f},{mem_state3:.1f},{mem_state4:.1f}")

    except Exception as e:
        print(f"error,{str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python3 {sys.argv[0]} [model_path] [num_threads] [num_images]")
        sys.exit(1)
    worker(sys.argv[1], int(sys.argv[3]), int(sys.argv[2]))
