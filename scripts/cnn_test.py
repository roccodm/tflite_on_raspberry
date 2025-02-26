# Released under CC0 License. Author: Rocco De Marco - CNR IRBIM Ancona  
#  
# This script evaluates the throughput and classification accuracy of a TFLite model  
# using a dataset of spectrogram images. It computes metrics such as true positives (TP),  
# false positives (FP), and inference speed (images/second).  
#  
# Usage: python3 cnn_test.py <model_path> <test_images_path> <num_thread>  
#   - model_path: Path to the TFLite model (e.g., "models/optimized_model_1.tflite").  
#   - test_images_path: Directory containing spectrogram images (e.g., "spectrogram/").  
#   - num_thread: Number of threads for the TFLite interpreter (1-8).  
#  
# Output: Throughput (images/second) and confusion matrix metrics, printed to the console.  

from PIL import Image
import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix
import glob
import tflite_runtime.interpreter as tflite
import time

# Constants
IMAGE_SIZE = (300, 150)
NFILES = 206

def get_positive_images_number(directory_path):   # Count the number of positive images in the directory.
    return len(glob.glob(f"{directory_path}/POS*.png"))

def split_indices(length, n):   # Split indices into blocks of size n.
    return [list(range(i, min(i + n, length))) for i in range(0, length, n)]


def load_images(directory_path, left_idx, right_idx):
    images, labels = [], []

    def load_and_preprocess(file_list, label):   #Helper function to load and preprocess images.
        for file_name in file_list:
            image = Image.open(file_name).convert("L")  # Convert to grayscale
            image = image.resize(IMAGE_SIZE)  # Resize
            images.append(np.array(image).astype(np.float32))
            labels.append(label)
    # Load positive images (label 1)
    positive_files = glob.glob(f"{directory_path}/POS*.png")[left_idx:right_idx]
    load_and_preprocess(positive_files, label=1)
    n_positives = len(positive_files)
    # Load negative images (label 0)
    negative_files = glob.glob(f"{directory_path}/NEG*.png")[left_idx:right_idx]
    load_and_preprocess(negative_files, label=0)
    print(f"*** Loaded {n_positives} positives and {len(negative_files)} negatives")
    return np.array(images) / 255, np.array(labels)

def run_test(model_path, x_test, y_test, num_thread): # Run inference using a specified TensorFlow Lite model.
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_thread)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    predicted_labels = []
    print(f"*** Starting tflite test for {len(x_test)} images")
    start = time.time()
    for i in range(len(x_test)):
        image = x_test[i].reshape(1, *IMAGE_SIZE, 1)  # Reshape for model input
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_labels.append(1 if prediction > 0.9 else 0)  # Binary classification
    end = time.time()
    p_time = end - start
    it_sec = len(x_test) / p_time
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, predicted_labels)
    tn, fp, fn, tp = conf_matrix.ravel()
    return tn, fp, fn, tp, len(x_test), p_time, it_sec

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: python {sys.argv[0]} <model_path> <test_images_path> <num_thread>")
        sys.exit(1)

    model_path, test_images_path, num_thread = sys.argv[1], sys.argv[2], int(sys.argv[3])

    n_positive = get_positive_images_number(test_images_path)
    image_blocks = split_indices(n_positive, NFILES)

    print ("*** Processing model:", model_path, "with", num_thread, "cores")
    i = 0
    for block in image_blocks:
        print(f"*** Loading images... (block {i})")
        if 'x_test' in locals():
            del(x_test)
        x_test, y_test = load_images(test_images_path, min(block), max(block) + 1)
        tn, fp, fn, tp, nimages, p_time, it_sec = run_test(model_path, x_test, y_test, num_thread)
        results = {"model": model_path, "block":i,  "tp":tp, "tn":tn, "fp":fp, "fn":fn, 
               "nimages":nimages, "process_time":p_time, "it_sec": it_sec}
        print (f"*** elapsed {nimages} images in {p_time} seconds ({it_sec}it/sec)")
        i += 1
        print ("*** raw results:", results)
