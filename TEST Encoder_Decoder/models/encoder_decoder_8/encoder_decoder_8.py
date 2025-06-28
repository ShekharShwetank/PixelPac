import tensorflow as tf
from tensorflow.keras import layers, models

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)


def image_to_feature_map(input_tensor):
    #conv to generate 3 feature map of same dimension
    x = layers.Conv2D(3, kernel_size = 3, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    # Depthwise + Pointwise (like MobileNet block)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.Conv2D(3, kernel_size=1, padding='same', activation='relu', name="A")(x)  # channel mixing
    
    return x

# def downsample_with_feature_enhancement(input_tensor):
#     # Initial regular conv to get richer early features
#     x = layers.Conv2D(8, kernel_size=3, strides=1, padding='same', use_bias=False)(input_tensor)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU(max_value=6)(x)

#     # Depthwise conv for spatial filtering and downsampling
#     x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU(max_value=6)(x)

#     # Pointwise conv to mix features and project to 8 channels
#     x = layers.Conv2D(8, kernel_size=1, padding='same', use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU(max_value=6, name = "B")(x)

#     return x


def increase_channels_3_to_8(input_tensor):
    # Conv to increase channels from 3 to 8, no downsampling (stride=1)
    x = layers.Conv2D(8, kernel_size=3, strides=1, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name = "B")(x)
    return x

def downsample_keep_channels(input_tensor):
    # Depthwise conv for spatial downsampling (stride=2)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Pointwise conv to mix features and keep channels same (8)
    x = layers.Conv2D(8, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="C")(x)
    return x

# def channel_expansion_with_feature_enhancement(input_tensor):
#     # Initial regular conv to extract richer features
#     x = layers.Conv2D(8, kernel_size=3, strides=1, padding='same', use_bias=False)(input_tensor)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU(max_value=6)(x)

#     # Depthwise conv for spatial filtering (no downsampling)
#     x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU(max_value=6)(x)

#     # Pointwise conv to mix and project to 16 channels
#     x = layers.Conv2D(16, kernel_size=1, padding='same', use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU(max_value=6, name = "D")(x)

#     return x

def channel_expansion_with_feature_enhancement(input_tensor):
    # Initial regular conv to extract richer features
    x = layers.Conv2D(8, kernel_size=3, strides=1, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Depthwise conv for spatial filtering (no downsampling)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Replace pointwise conv with heavy conv block to project to 16 channels
    x = layers.Conv2D(16, kernel_size=5, padding='same', use_bias=False)(x)  # heavier conv with larger kernel
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="D")(x)

    return x


def downsample_preserve_channels(input_tensor):
    # Optional initial conv to refine features before downsampling
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Depthwise convolution with stride=2 for downsampling
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Pointwise conv to mix features and preserve 16 channels
    x = layers.Conv2D(16, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name = "E")(x)

    return x

def channel_expand_by_2x(input_tensor):
    # Initial standard conv for richer feature extraction
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Depthwise conv to filter spatial information
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Pointwise conv to mix and expand channels from 16 → 32
    x = layers.Conv2D(32, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name = "F")(x)

    return x

def downsample_by_2(input_tensor):
    # Optional initial conv to refine features before downsampling
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Depthwise convolution with stride=2 for downsampling by factor of 2
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Pointwise conv to mix features and preserve 32 channels
    x = layers.Conv2D(32, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="G")(x)

    return x

def channel_expansion_32_to_64(input_tensor):
    # Initial regular conv to extract richer features
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Depthwise conv for spatial filtering (no downsampling)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Heavier conv block to project to 64 channels with 4x4 kernel
    x = layers.Conv2D(64, kernel_size=4, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name = "H")(x)

    return x

def downsample_by_2_keep_channels(input_tensor):
    channels = input_tensor.shape[-1]

    # Optional initial conv to refine features before downsampling
    x = layers.Conv2D(channels, kernel_size=3, strides=1, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Depthwise convolution with stride=2 to downsample spatial dimensions by 2
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Pointwise conv to preserve the same number of channels
    x = layers.Conv2D(channels, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name = "I")(x)

    return x


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
        print(f"Inference output shape = {_.shape}, time = {inference_time:.2f} ms")

    avg_time = sum(times) / len(times)
    print(f"Average inference time over {num_runs} runs: {avg_time:.6f} seconds")

import os

def save_model():
    save_path = os.path.join(os.path.dirname(__file__), "model.h5")
    model.save(save_path)
    print(f"Model saved at: {save_path}")

# # Example usage
# if __name__ == "__main__":
#     skip_connections = []
#     inputs = layers.Input(shape=(176, 608, 3))
#     outputs = image_to_feature_map(inputs)
    
#     outputs = increase_channels_3_to_8(outputs)
#     skip_connections.append(outputs)

#     outputs = downsample_keep_channels(outputs)

#     outputs = channel_expansion_with_feature_enhancement(outputs)
#     skip_connections.append(outputs)

#     outputs = downsample_preserve_channels(outputs)
#     outputs = channel_expand_by_2x(outputs)
#     skip_connections.append(outputs)

#     outputs = downsample_by_2(outputs)
#     outputs = channel_expansion_32_to_64(outputs)
#     skip_connections.append(outputs)

#     outputs = downsample_by_2_keep_channels(outputs)
#     model = models.Model(inputs=inputs, outputs=outputs)
#     model.summary()

#     dummy = tf.random.normal([1, 176, 608, 3])
#     out = model(dummy)
#     print("Output shape:", out.shape)  # Expected: (1, 176, 608, 3)

#     # Define or load your model and input_tensor above this block
#     measure_inference_time(model, dummy)


def build_model(input_shape=(176, 608, 3)):
    inputs = layers.Input(shape=input_shape)
    skip_connections = []

    x = image_to_feature_map(inputs)

    x = increase_channels_3_to_8(x)
    skip_connections.append(x)

    x = downsample_keep_channels(x)

    x = channel_expansion_with_feature_enhancement(x)
    skip_connections.append(x)

    x = downsample_preserve_channels(x)

    x = channel_expand_by_2x(x)
    skip_connections.append(x)

    x = downsample_by_2(x)

    x = channel_expansion_32_to_64(x)
    skip_connections.append(x)

    x = downsample_by_2_keep_channels(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model

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
def run_tflite_inference(tflite_model_path, input_shape=(1, 176, 608, 3), num_runs=10):
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
    model = build_model()
    model.summary()

    dummy = tf.random.normal([1, 176, 608, 3])
    out = model(dummy)
    print("Output shape:", out.shape)  # Should be (1, 88, 304, 64)

    # Define or load your model and input_tensor above this block
    measure_inference_time(model, dummy)

    # save_model()

    # ################################### Convert Model to TFLite ######################################################
    # input_shape = (176, 608, 3)
    # keras_model = build_model(input_shape=input_shape)
    # tflite_model_path = convert_to_tflite(keras_model)

    # run_tflite_inference(tflite_model_path)