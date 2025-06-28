import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, UpSampling2D,
    Concatenate, Lambda, Input
)
from tensorflow.keras.models import Model
import os
import numpy as np
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)


import time
def build_model(input_shape=(240, 240, 3)):
    inputs = Input(shape=input_shape)
    intermediate_features = []

    # Expand: 3 -> 48
    x = Conv2D(48, kernel_size=3, strides=1, padding='same', name='conv_expand')(inputs)
    x = BatchNormalization(name='bn_expand')(x)
    x = ReLU(name='relu_expand')(x)

    # Downsample: H, W -> H/2, W/2
    x = Conv2D(48, kernel_size=3, strides=2, padding='same', name='conv_downsample')(x)
    x = BatchNormalization(name='bn_downsample')(x)
    x = ReLU(name='LAYER_1_OUT')(x)

    # Identity Conv Block
    x = Conv2D(48, kernel_size=3, strides=1, padding='same', name='conv_identity')(x)
    x = BatchNormalization(name='bn_identity')(x)
    x = ReLU(name='LAYER_2_OUT')(x)

    # Reduce: 48 -> 32
    x = Conv2D(32, kernel_size=1, strides=1, padding='same', name='conv_reduce')(x)
    x = BatchNormalization(name='bn_reduce')(x)
    x = ReLU(name='LAYER_4_OUT')(x)
    intermediate_features.append(x)

    # Increase: 32 -> 40
    x = Conv2D(40, kernel_size=1, strides=1, padding='same', name='conv_increase')(x)
    x = BatchNormalization(name='bn_increase')(x)
    x = ReLU(name='relu_increase')(x)

    # Downsample
    x = Conv2D(40, kernel_size=3, strides=2, padding='same', name='conv_downsample_40')(x)
    x = BatchNormalization(name='bn_downsample_40')(x)
    x = ReLU(name='LAYER_5_OUT')(x)
    intermediate_features.append(x)

    # Increase: 40 -> 64
    x = Conv2D(64, kernel_size=1, strides=1, padding='same', name='conv_increase_64')(x)
    x = BatchNormalization(name='bn_increase_64')(x)
    x = ReLU(name='relu_increase_64')(x)

    # Downsample
    x = Conv2D(64, kernel_size=3, strides=2, padding='same', name='conv_downsample_64')(x)
    x = BatchNormalization(name='bn_downsample_64')(x)
    x = ReLU(name='LAYER_6_OUT')(x)
    intermediate_features.append(x)

    # Increase: 64 -> 128
    x = Conv2D(128, kernel_size=1, strides=1, padding='same', name='conv_increase_128')(x)
    x = BatchNormalization(name='bn_increase_128')(x)
    x = ReLU(name='relu_increase_128')(x)

    # Downsample
    x = Conv2D(128, kernel_size=3, strides=2, padding='same', name='conv_downsample_128')(x)
    x = BatchNormalization(name='bn_downsample_128')(x)
    x = ReLU(name='LAYER_7_OUT')(x)

    # Additional Downsample
    x = Conv2D(128, kernel_size=3, strides=2, padding='same', name='conv_downsample_identity')(x)
    x = BatchNormalization(name='bn_downsample_identity')(x)
    x = ReLU(name='LAYER_8_OUT')(x)
    intermediate_features.append(x)

    # Unpack features
    if len(intermediate_features) != 4:
        raise ValueError("Expected 4 intermediate features but got {}".format(len(intermediate_features)))

    xd_0, xd_1, xd_2, xd_3 = intermediate_features

    # Process xd_3
    x3 = UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_xd_3')(xd_3)
    x3 = Conv2D(128, kernel_size=3, padding='same', name='conv_xd_3_upsampled')(x3)
    x3 = BatchNormalization(name='bn_xd_3_upsampled')(x3)
    x3 = ReLU(name='relu_xd_3_upsampled')(x3)
    x3 = Conv2D(96, kernel_size=1, padding='same', name='conv_xd_3_reduce')(x3)
    x3 = BatchNormalization(name='bn_xd_3_reduce')(x3)
    x3 = ReLU(name='relu_xd_3_reduce')(x3)
    x3 = UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_xd_3_to_75')(x3)
    x3 = BatchNormalization(name='bn_xd_3_final')(x3)
    x3 = ReLU(name='relu_xd_3_final')(x3)
    x3 = Lambda(lambda x: tf.image.resize(x, (75, 75)), name='resize_xd_3_to_75')(x3)

    # Fuse xd_3 and xd_2
    fused = Concatenate(name='concat_xd_3_2')([x3, xd_2])
    fused = Conv2D(64, kernel_size=3, padding='same', name='conv_fused_xd')(fused)
    fused = BatchNormalization(name='bn_fused_xd')(fused)
    fused = ReLU(name='relu_fused_xd')(fused)

    fused = UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_fused_xd')(fused)
    fused = Conv2D(128, kernel_size=3, padding='same', name='conv_after_upsample')(fused)
    fused = BatchNormalization(name='bn_after_upsample')(fused)
    fused = ReLU(name='relu_after_upsample')(fused)

    fused = Concatenate(name='concat_fused_xd_xd1')([fused, xd_1])
    fused = Conv2D(128, kernel_size=3, padding='same', name='conv_fused_xd_128')(fused)
    fused = BatchNormalization(name='bn_fused_xd_128')(fused)
    fused = ReLU(name='relu_fused_xd_128')(fused)

    fused = UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_fused_xd_2')(fused)
    fused = Conv2D(128, kernel_size=3, padding='same', name='conv_after_upsample_2')(fused)
    fused = BatchNormalization(name='bn_after_upsample_2')(fused)
    fused = ReLU(name='relu_after_upsample_2')(fused)

    fused = Concatenate(name='concat_fused_xd_xd0')([fused, xd_0])
    fused = Conv2D(64, kernel_size=3, padding='same', name='conv_fused_xd_64')(fused)
    fused = BatchNormalization(name='bn_fused_xd_64')(fused)
    fused = ReLU(name='relu_fused_xd_64')(fused)

    model = Model(inputs, fused, name='CustomFeatureFusionNet')
    return model, intermediate_features

def measure_inference_time(model, input_shape, num_runs=10):
    times = []
    
    # Run a warm-up inference
    model, intermediante_featues = build_model(input_shape=input_shape)

    for _ in range(num_runs):
        start = time.time()
        _ = model(input_tensor)
        end = time.time()
        times.append(end - start)
        inference_time = (end - start) * 1000  # milliseconds
        print(f"Inference output shape = {_.shape}, time = {inference_time:.2f} ms")

    avg_time = sum(times) / len(times)
    print(f"Average inference time over {num_runs} runs: {avg_time:.6f} seconds")


# def measure_inference_time(model, input_tensor, num_runs=10):
#     times = []

#     # Warm-up
#     modell , intermediante_featues= build_model(input_tensor)

#     for _ in range(num_runs):
#         start = time.time()
#         _ = modell(input_tensor)
#         end = time.time()
#         times.append(end - start)

#     avg_time = sum(times) / len(times)
#     print(f"Average inference time over {num_runs} runs: {avg_time:.6f} seconds")

def save_model():
    input_shape = (600, 600, 3)
    model, intermediante_featues = build_model(input_shape=input_shape)
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
    # Build and summarize the model
    input_shape = (600, 600, 3)
    model, intermediante_featues = build_model(input_shape=input_shape)
    
    input_tensor = tf.random.normal([1,600, 600, 3])
    output = model(input_tensor)
    model.summary()

    # Define or load your model and input_tensor above this block
    measure_inference_time(model, input_shape)

    # save_model()

    # #################################### Convert Model to TFLite ######################################################
    # input_shape = (600, 600, 3)
    # keras_model, _ = build_model(input_shape=input_shape)
    # tflite_model_path = convert_to_tflite(keras_model)

    # run_tflite_inference(tflite_model_path)