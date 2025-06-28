import tensorflow as tf
import numpy as np
import time

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="sample_untrained_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
for i in range(20):
# Prepare input data (random, matching model's expected input shape)
    input_data = np.random.normal(size=input_shape).astype(np.float32)

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference and measure time
    start_time = time.time()
    interpreter.invoke()
    elapsed_time = time.time() - start_time

    # Get output data
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(f"Inference time: {elapsed_time * 1000:.2f} ms")
    print("Output shape:", output_data.shape)
