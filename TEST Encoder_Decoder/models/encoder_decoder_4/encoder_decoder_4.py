import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Lambda
import time
import os

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

def FastDepth(input_shape=(600, 600, 3), output_channels=64):
    inputs = tf.keras.Input(shape=input_shape)

    # MobileNetV2 backbone with input tensor
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights='imagenet'
    )

    # Encoder layers for skip connections
    encoder_output = base_model.get_layer('block_13_expand_relu').output  # [38x38x576]
    skip_1 = base_model.get_layer('block_6_expand_relu').output           # [75x75x192]
    skip_2 = base_model.get_layer('block_3_expand_relu').output           # [150x150x96]
    skip_3 = base_model.get_layer('block_1_expand_relu').output           # [150x150x24]

    # Decoder block 1: 38x38 → 75x75
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(encoder_output)  # [75x75x256]
    x = Lambda(lambda x: tf.image.resize(x, [75, 75]))(x)
    x = layers.Concatenate()([x, skip_1])
    x = layers.DepthwiseConv2D(3, padding='same')(x)
    x = layers.Conv2D(128, 1, padding='same', activation='relu')(x)

    # Decoder block 2: 75x75 → 150x150
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)  # [150x150x128]
    x = layers.Concatenate()([x, skip_2])
    x = layers.DepthwiseConv2D(3, padding='same')(x)
    x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)

    # Decoder block 3: 150x150 → 300x300
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)  # [300x300x64]
    x = layers.Concatenate()([x, skip_3])
    x = layers.DepthwiseConv2D(3, padding='same')(x)
    x = layers.Conv2D(output_channels, 1, padding='same', activation='relu')(x)


    return models.Model(inputs, x, name="FastDepth_Tiny")


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
        print(f"Inference output shape = {_.shape}, time = {inference_time:.2f} ms")

    avg_time = sum(times) / len(times)
    print(f"Average inference time over {num_runs} runs: {avg_time:.6f} seconds")

def save_model():
    model = FastDepth()
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
def run_tflite_inference(tflite_model_path, input_shape=(1, 600, 600, 3), num_runs=10):
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


if __name__ == "__main__":
    # Build and summarize
    model = FastDepth()
    model.summary()
    dummy = tf.random.normal([1,600, 600, 3])
    # Define or load your model and input_tensor above this block
    measure_inference_time(model, dummy)

    # save_model()

    #################################### Convert Model to TFLite ######################################################
    # input_shape = (600, 600, 3)
    # keras_model = FastDepth(input_shape=input_shape)
    # tflite_model_path = convert_to_tflite(keras_model)

    # run_tflite_inference(tflite_model_path)
