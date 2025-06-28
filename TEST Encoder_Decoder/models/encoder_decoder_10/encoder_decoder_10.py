from tensorflow.keras import layers, models
import tensorflow as tf

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

def image_to_feature_map_modified(input_tensor):
    # Use 3x3 depthwise separable conv instead of 5x5
    x = layers.SeparableConv2D(8, kernel_size=3, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="A")(x)
    return x

def downsample_keep_channels(input_tensor):
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.Conv2D(8, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="B")(x)
    return x

def feature_map_8_to_16(input_tensor):
    x = layers.SeparableConv2D(16, kernel_size=3, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="C")(x)
    return x

def downsample_keep_channels_modified(input_tensor):
    channels = input_tensor.shape[-1]
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.Conv2D(channels, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="D")(x)
    return x

def channel_expand_by_2x(input_tensor):
    x = layers.SeparableConv2D(32, kernel_size=3, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="E")(x)
    return x

def downsample_by_2(input_tensor):
    x = layers.SeparableConv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="F")(x)
    return x

def feature_map_32_to_64(input_tensor):
    x = layers.SeparableConv2D(64, kernel_size=3, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="G")(x)
    return x

def downsample_by_2_keep_channels(input_tensor):
    channels = input_tensor.shape[-1]
    x = layers.SeparableConv2D(channels, kernel_size=3, strides=2, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="H")(x)
    return x

def channel_expand_64_to_128(input_tensor):
    x = layers.SeparableConv2D(128, kernel_size=3, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="I")(x)
    return x


def fuse_and_project_features(x1, x2):
    # x1: (None, 22, 76, 64)
    # x2: (None, 11, 38, 128)

    # Upsample x2 to match x1's spatial dimensions
    x2_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x2)  # (None, 22, 76, 128)

    # Project x2 to 64 channels to match x1
    x2_projected = layers.Conv2D(64, kernel_size=1, padding='same', use_bias=False)(x2_upsampled)
    x2_projected = layers.BatchNormalization()(x2_projected)
    x2_projected = layers.ReLU(max_value=6)(x2_projected)

    # Concatenate along the channel axis: (None, 22, 76, 64 + 64)
    x = layers.Concatenate(axis=-1)([x1, x2_projected])  # (None, 22, 76, 128)

    # Fuse features and project back to 64 channels
    x = layers.Conv2D(64, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="J")(x)

    return x


def fuse_features_to_32(x1, x2):
    # x1: (None, 44, 152, 32)
    # x2: (None, 22, 76, 64)

    # Step 1: Upsample x2 to match x1's spatial dimensions
    x2_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x2)  # (None, 44, 152, 64)

    # Step 2: Project x2 to 32 channels
    x2_projected = layers.Conv2D(32, kernel_size=1, padding='same', use_bias=False)(x2_upsampled)
    x2_projected = layers.BatchNormalization()(x2_projected)
    x2_projected = layers.ReLU(max_value=6)(x2_projected)

    # Step 3: Concatenate x1 and x2_projected → (None, 44, 152, 64)
    x = layers.Concatenate(axis=-1)([x1, x2_projected])

    # Step 4: Fuse and reduce to 32 channels
    x = layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="K")(x)

    return x


def fuse_features_to_16(x1, x2):
    # x1: (None, 88, 304, 16)
    # x2: (None, 44, 152, 32)

    # Step 1: Upsample x2 to match x1's spatial dimensions
    x2_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x2)  # (None, 88, 304, 32)

    # Step 2: Project x2 to 16 channels
    x2_projected = layers.Conv2D(16, kernel_size=1, padding='same', use_bias=False)(x2_upsampled)
    x2_projected = layers.BatchNormalization()(x2_projected)
    x2_projected = layers.ReLU(max_value=6)(x2_projected)

    # Step 3: Concatenate x1 and x2_projected → (None, 88, 304, 32)
    x = layers.Concatenate(axis=-1)([x1, x2_projected])

    # Step 4: Fuse and reduce to 16 channels
    x = layers.Conv2D(16, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="L")(x)

    return x


def fuse_3_16_to_64(x1, x2):
    # x1: (None, 176, 608, 3)
    # x2: (None, 88, 304, 16)

    # Step 1: Upsample x2 to match x1's spatial dimensions
    x2_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x2)  # (None, 176, 608, 16)

    # Step 2: Project x1 to 32 channels
    x1_proj = layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False)(x1)
    x1_proj = layers.BatchNormalization()(x1_proj)
    x1_proj = layers.ReLU(max_value=6)(x1_proj)

    # Step 3: Project x2 to 32 channels
    x2_proj = layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False)(x2_upsampled)
    x2_proj = layers.BatchNormalization()(x2_proj)
    x2_proj = layers.ReLU(max_value=6)(x2_proj)

    # Step 4: Concatenate both → (None, 176, 608, 64)
    x = layers.Concatenate(axis=-1)([x1_proj, x2_proj])

    # Step 5: Fuse and output 64 channels
    x = layers.Conv2D(64, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="output")(x)

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
    # model = Custom_Encoder_Decoder()
    save_path = os.path.join(os.path.dirname(__file__), "model.h5")
    model.save(save_path)
    print(f"Model saved at: {save_path}")

def build_model(input_shape ):
    inputs = layers.Input(shape=input_shape, name="input_image")
    skip_connections = []

    # Encoder
    x = image_to_feature_map_modified(inputs)  # Output: (176, 608, 8)
    skip_connections.append(x)  # A

    x = downsample_keep_channels(x)            # (88, 304, 8)
    x = feature_map_8_to_16(x)                 # (88, 304, 16)
    skip_connections.append(x)  # C

    x = downsample_keep_channels_modified(x)   # (44, 152, 16)
    x = channel_expand_by_2x(x)                # (44, 152, 32)
    skip_connections.append(x)  # E

    x = downsample_by_2(x)                     # (22, 76, 32)
    x = feature_map_32_to_64(x)                # (22, 76, 64)
    skip_connections.append(x)  # G

    x = downsample_by_2_keep_channels(x)       # (11, 38, 64)
    x = channel_expand_64_to_128(x)            # (11, 38, 128)

    # Decoder with Fusion
    x = fuse_and_project_features(skip_connections[3], x)      # (22, 76, 64)
    x = fuse_features_to_32(skip_connections[2], x)            # (44, 152, 32)
    x = fuse_features_to_16(skip_connections[1], x)            # (88, 304, 16)
    x = fuse_3_16_to_64(inputs, x)                             # (176, 608, 64)

    outputs = x  # Final output tensor (or use Conv2D for regression/classification if needed)

    model = models.Model(inputs=inputs, outputs=outputs, name="Custom_Encoder_Decoder")
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
    input_shape = (176, 608, 3)
    model = build_model(input_shape = input_shape)
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