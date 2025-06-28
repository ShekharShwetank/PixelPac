import tensorflow as tf
from tensorflow.keras import layers, models

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

def image_to_feature_map_modified(input_tensor):
    # Conv to increase channels from 3 to 8 using 5x5 kernel
    x = layers.Conv2D(8, kernel_size=5, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)

    # Depthwise + Pointwise
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Channel mixing with Conv 1x1 (keeps channels at 8)
    x = layers.Conv2D(8, kernel_size=1, padding='same', activation='relu', name="A")(x)
    
    return x

def downsample_keep_channels(input_tensor):
    # Depthwise conv for spatial downsampling (stride=2)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Pointwise conv to mix features and keep channels same (8)
    x = layers.Conv2D(8, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="B")(x)
    return x

def feature_map_8_to_16(input_tensor):
    # Conv to increase channels from 8 to 16 using 5x5 kernel
    x = layers.Conv2D(16, kernel_size=5, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)

    # Depthwise + Pointwise
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Channel mixing with Conv 1x1 (keeps channels at 16)
    x = layers.Conv2D(16, kernel_size=1, padding='same', activation='relu', name="C")(x)
    
    return x

def downsample_keep_channels_modified(input_tensor):
    # Depthwise conv for spatial downsampling (stride=2)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Pointwise conv to keep the number of channels unchanged
    channels = input_tensor.shape[-1]
    x = layers.Conv2D(channels, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6, name="D")(x)
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
    x = layers.ReLU(max_value=6, name = "E")(x)

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
    x = layers.ReLU(max_value=6, name="F")(x)

    return x

def feature_map_32_to_64(input_tensor):
    # Conv to increase channels from 32 to 64 using 4x4 kernel
    x = layers.Conv2D(64, kernel_size=4, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)

    # Depthwise + Pointwise
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Channel mixing with Conv 1x1 (keeps channels at 64)
    x = layers.Conv2D(64, kernel_size=1, padding='same', activation='relu', name="G")(x)
    
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
    x = layers.ReLU(max_value=6, name = "H")(x)

    return x


def channel_expand_64_to_128(input_tensor):
    # Initial standard conv for richer feature extraction
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Depthwise conv to filter spatial information
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Pointwise conv to mix and expand channels from 64 → 128
    x = layers.Conv2D(128, kernel_size=1, padding='same', use_bias=False)(x)
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

def build_model(input_shape=(176, 608, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    # Stage 1: Initial feature extraction (→ (176, 608, 8))
    x1 = image_to_feature_map_modified(inputs)

    # Stage 2: Downsample (→ (88, 304, 8))
    x2 = downsample_keep_channels(x1)

    # Stage 3: Expand channels to 16 (→ (88, 304, 16))
    x2 = feature_map_8_to_16(x2)

    # Stage 4: Downsample (→ (44, 152, 16))
    x3 = downsample_keep_channels_modified(x2)

    # Stage 5: Expand channels to 32 (→ (44, 152, 32))
    x3 = channel_expand_by_2x(x3)

    # Stage 6: Downsample (→ (22, 76, 32))
    x4 = downsample_by_2(x3)

    # Stage 7: Expand to 64 (→ (22, 76, 64))
    x4 = feature_map_32_to_64(x4)

    # Stage 8: Downsample (→ (11, 38, 64))
    x5 = downsample_by_2_keep_channels(x4)

    # Stage 9: Expand to 128 (→ (11, 38, 128))
    x5 = channel_expand_64_to_128(x5)

    # Fusion 1: (x4: 22x76x64, x5: 11x38x128) → (22x76x64)
    x = fuse_and_project_features(x4, x5)

    # Fusion 2: (x3: 44x152x32, x: 22x76x64) → (44x152x32)
    x = fuse_features_to_32(x3, x)

    # Fusion 3: (x2: 88x304x16, x: 44x152x32) → (88x304x16)
    x = fuse_features_to_16(x2, x)

    # Final upsampling to match input size (176x608)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # (176, 608, 16)

    # Output projection to 1 channel (e.g., for depth estimation or segmentation mask)
    outputs = layers.Conv2D(1, kernel_size=1, activation='sigmoid', name="final_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientDepthNet")
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
    