# TinyML Model Performance Testing Suite

This repository contains:
- three Python scripts designed to evaluate the performance of TensorFlow Lite models deployed on embedded devices such as the Raspberry Pi Zero 2W. These scripts assess different aspects of model execution, including memory usage, inference throughput, and latency.
- 1646 spectrogram images, 823 of them (which filename starting with POS) contains a significative portiono of a dolphin whistle
- 2 trained CNN TFlite models for whistle detection: optimized and non-optimized

## **1. Memory Usage Test** (`memory_test.py`)

### **Description**
This script evaluates the memory consumption of a TensorFlow Lite model at different execution stages, including:
- Model loading
- Tensor allocation
- Inference execution

### **Usage**
```bash
python3 memory_test.py <model_path> <num_threads> <num_images>
```

### **Parameters**
- `<model_path>`: Path to the TensorFlow Lite model (e.g., `models/optimized_model_1.tflite`).
- `<num_threads>`: Number of threads allocated for inference execution.
- `<num_images>`: Number of test images generated for inference.

### **Example**
```bash
python3 memory_test.py models/optimized_model_1.tflite 4 100
```

---

## **2. Throughput Test** (`cnn_test.py`)

### **Description**
This script measures the inference throughput of a CNN model by processing spectrogram images stored in a directory. It calculates classification performance metrics such as True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).

### **Usage**
```bash
python3 cnn_test.py <model_path> <test_images_path> <num_threads>
```

### **Parameters**
- `<model_path>`: Path to the TensorFlow Lite model (e.g., `models/optimized_model_1.tflite`).
- `<test_images_path>`: Directory containing test spectrograms (e.g., `spectrogram/`).
- `<num_threads>`: Number of threads allocated for inference execution.

### **Example**
```bash
python3 cnn_test.py models/optimized_model_1.tflite spectrogram/ 4
```

---

## **3. Latency Test** (`latency_test.py`)

### **Description**
This script measures the inference latency of a TensorFlow Lite model across different thread configurations (from 1 to 8 threads). The script performs multiple inference runs and computes statistical metrics such as:
- Average latency
- Minimum and maximum latency
- Standard deviation

### **Usage**
```bash
python3 latency_test.py <model_path> <num_runs>
```

### **Parameters**
- `<model_path>`: Path to the TensorFlow Lite model (e.g., `models/optimized_model_1.tflite`).
- `<num_runs>`: Number of inference executions for latency measurement.

### **Example**
```bash
python3 latency_test.py models/optimized_model_1.tflite 100
```

---

## **General Notes**
- Ensure that all dependencies (e.g., `numpy`, `tflite_runtime`, `PIL`, `sklearn`, `psutil`) are installed.
- Spectrogram images should be placed inside `spectrogram/` and follow the naming convention (`POS*.png` for positive samples and `NEG*.png` for negative samples).
- Results are printed to the console and include key performance indicators for analysis.

