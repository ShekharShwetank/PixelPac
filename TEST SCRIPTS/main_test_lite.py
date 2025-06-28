import os
import time
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs

def load_tflite_model(model_path):
    """Loads and allocates a TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    return interpreter

def run_tflite_inference(model_path, num_runs=10):
    """Runs inference on a TFLite model and prints timing."""
    interpreter = load_tflite_model(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    times = []

    # Warm-up
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    for i in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        start = time.time()
        interpreter.invoke()
        end = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        elapsed = (end - start) * 1000  # milliseconds
        times.append(elapsed)

        print(f"[{i+1}] Output shape: {output_data.shape}, Time: {elapsed:.2f} ms")

    avg_time = sum(times) / len(times)
    print(f"✅ {os.path.basename(model_path)}: Average inference time over {num_runs} runs: {avg_time:.2f} ms\n")

if __name__ == "__main__":
    model_dir = "lite"
    model_files = sorted([
        f for f in os.listdir(model_dir)
        if f.endswith(".tflite") and os.path.isfile(os.path.join(model_dir, f))
    ])
    print(model_files)
    for model_file in model_files:
        
        model_path = os.path.join(model_dir, model_file)
        print(f"🔍 Running inference on: {model_file}")
        run_tflite_inference(model_path)