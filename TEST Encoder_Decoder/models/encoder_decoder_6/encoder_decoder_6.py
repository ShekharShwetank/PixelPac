import tensorflow as tf
from tensorflow.keras import layers, models
import os

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

def build_encoder(input_shape=(480, 480, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights='imagenet'
    )

    # Feature maps at different scales:
    # You can inspect MobileNetV2 layers to find the ones matching desired output sizes
    
    # For 480 input, these layers output approx:
    # block_1_expand_relu  -> 240x240x24
    # block_3_expand_relu  -> 120x120x96
    # block_6_expand_relu  -> 60x60x192
    # block_13_expand_relu -> 30x30x576

    feat_240 = base_model.get_layer('block_1_expand_relu').output    # ~240x240x24
    feat_120 = base_model.get_layer('block_3_expand_relu').output    # ~120x120x96
    feat_60  = base_model.get_layer('block_6_expand_relu').output    # ~60x60x192
    feat_30  = base_model.get_layer('block_13_expand_relu').output   # ~30x30x576

    return models.Model(inputs=inputs, outputs=[feat_240, feat_120, feat_60, feat_30], name="Encoder_480")


import time
def measure_inference_time(model, input_tensor, num_runs=10):
    times = []

    # Run a warm-up inference
    _ = model(input_tensor)

    for _ in range(num_runs):
        start = time.time()
        _ = model(input_tensor)
        end = time.time()
        times.append(end - start)

        inference_time = (end - start) * 1000  # milliseconds
        print(f"Inference time = {inference_time:.2f} ms")

    avg_time = sum(times) / len(times)
    print(f"Average inference time over {num_runs} runs: {avg_time:.6f} seconds")



def save_model():
    model = build_encoder()
    save_path = os.path.join(os.path.dirname(__file__), "model.h5")
    model.save(save_path)
    print(f"Model saved at: {save_path}")



def convert_to_tflite(keras_model, tflite_filename='encoder_decoder.tflite'):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tflite_model_path = os.path.join(script_dir, tflite_filename)
    
    # Create a converter object from the Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # Optional optimizations (uncomment if needed)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the converted model to disk
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {tflite_model_path}")
    return tflite_model_path


import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO (1), WARNING (2), and ERROR (3) logs
def run_tflite_inference(tflite_model_path, input_shape=(1, 480, 480, 3), num_runs=10):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create dummy input tensor
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
    print(f"✅ Average inference time over {num_runs} runs: {avg_time:.2f} ms")
    

# Example usage:
if __name__ == "__main__":
    encoder = build_encoder()
    encoder.summary()

    dummy_input = tf.random.normal([1, 480, 480, 3])
    feat_240, feat_120, feat_60, feat_30 = encoder(dummy_input)

    # print(f"240x240 feature map shape: {feat_240.shape}")
    # print(f"120x120 feature map shape: {feat_120.shape}")
    # print(f"60x60 feature map shape: {feat_60.shape}")
    # print(f"30x30 feature map shape: {feat_30.shape}")

    # Define or load your model and input_tensor above this block
    measure_inference_time(encoder, dummy_input)

    # save_model()

    # ################################### Convert Model to TFLite ######################################################
    # input_shape = (480, 480, 3)
    # keras_model = build_encoder(input_shape=input_shape)
    # tflite_model_path = convert_to_tflite(keras_model)

    # run_tflite_inference(tflite_model_path)