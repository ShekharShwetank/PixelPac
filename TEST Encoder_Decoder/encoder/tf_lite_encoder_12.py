import time
import tensorflow as tf
import numpy as np

def run_tflite_inference_with_timing(tflite_model_path='encoder_12.tflite', input_shape=(176,304,3), num_iterations=10):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path, num_threads=4)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    for i in range(num_iterations):
        input_data = np.random.random_sample((1,) + input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        inference_time = (end_time - start_time) * 1000  # milliseconds
        print(f"Inference {i+1}: output shape = {output_data.shape}, time = {inference_time:.2f} ms")

# Run inference with timing
run_tflite_inference_with_timing()