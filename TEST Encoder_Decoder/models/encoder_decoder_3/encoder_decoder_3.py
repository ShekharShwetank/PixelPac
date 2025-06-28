import tensorflow as tf
from tensorflow.keras.layers import (
    Input, SeparableConv2D, BatchNormalization, LeakyReLU, Conv2D, Lambda,
    Conv2DTranspose, Concatenate, Lambda, MaxPooling2D
)
from tensorflow.keras.models import Model
import time
import os

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)


def build_optimized_model(input_shape=(240, 240, 3)):
    inputs = Input(shape=input_shape)
    intermediate_features = []
    
    # Initial Block with Separable Convolution
    x = SeparableConv2D(32, 3, strides=1, padding='same', name='sep_conv_expand')(inputs)
    x = BatchNormalization(name='bn_expand')(x, training=False)  # Correct placement
    x = LeakyReLU(alpha=0.1, name='lrelu_expand')(x)

    # Downsample Block 1
    x = SeparableConv2D(48, 3, strides=2, padding='same', name='sep_conv_down1')(x)
    x = BatchNormalization(name='bn_down1')(x, training=False)  # Moved training flag
    x = LeakyReLU(alpha=0.1, name='LAYER_1_OUT')(x)
    
    # Identity Block
    residual = x
    x = SeparableConv2D(48, 3, padding='same', name='sep_conv_identity')(x)
    x = BatchNormalization(name='bn_identity')(x, training=False)  # Corrected
    x = LeakyReLU(alpha=0.1, name='LAYER_2_OUT')(x)
    x = tf.keras.layers.add([x, residual])

    # Channel Reduction Block
    x = SeparableConv2D(32, 1, padding='same', name='sep_conv_reduce')(x)
    x = BatchNormalization(name='bn_reduce')(x, training=False)  # Fixed
    x = LeakyReLU(alpha=0.1, name='LAYER_4_OUT')(x)
    intermediate_features.append(x)
    
    # Efficient Downsampling Path
    # Efficient Downsampling Path
    for filters in [40, 64, 128]:
        x = Conv2D(filters, 3, strides=2, padding='same')(x)  # Downsampling
        x = BatchNormalization()(x, training=False)
        x = LeakyReLU(alpha=0.1)(x)
        intermediate_features.append(x)

    

    # Feature Fusion Network
    xd_0, xd_1, xd_2, xd_3 = intermediate_features
    # print(f"xd_1 shape is : {xd_0.shape}")
    # print(f"xd_0 shape is : {xd_1.shape}")
    # print(f"xd_2 shape is : {xd_2.shape}")
    # print(f"xd_3 shape is : {xd_3.shape}")

    # Channel Reduction Before Fusion
    xd_3 = Conv2D(64, 1, padding='same', name='channel_reduce_xd3')(xd_3)
    xd_2 = Conv2D(64, 1, padding='same', name='channel_reduce_xd2')(xd_2)
    # Resize xd_2 using a Lambda layer
    xd_2_resized = Lambda(lambda img: tf.image.resize(img, size=(76, 76)))(xd_2)

    # print(f"xd_3 :{xd_3.shape}")
    # print(f"xd_2 :{xd_2.shape}")

    fused_xd = Concatenate(axis=-1)([
        Conv2DTranspose(64, 3, strides=2, padding='same')(xd_3), 
        xd_2_resized
    ])
    fused_xd = Lambda(lambda x: tf.image.resize(x, size=(75, 75)))(fused_xd)
    # print(f"fused_xd shape is {fused_xd.shape}")
    fused_xd = SeparableConv2D(128, 3, padding='same')(fused_xd)
    fused_xd = BatchNormalization()(fused_xd, training=False)
    fused_xd = LeakyReLU(alpha=0.1)(fused_xd)

    # Progressive Fusion
    for feat in [xd_1, xd_0]:
        fused_xd = Conv2DTranspose(128, 3, strides=2, padding='same')(fused_xd)
        fused_xd = Concatenate(axis=-1)([fused_xd, feat])
        fused_xd = SeparableConv2D(128, 3, padding='same')(fused_xd)
        fused_xd = BatchNormalization()(fused_xd, training=False)
        fused_xd = LeakyReLU(alpha=0.1)(fused_xd)

    # Final Projection
    fused_xd = SeparableConv2D(64, 3, padding='same')(fused_xd)
    fused_xd = BatchNormalization()(fused_xd, training=False)
    fused_xd = LeakyReLU(alpha=0.1)(fused_xd)

    model = Model(inputs, fused_xd)
    return model, intermediate_features

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
    model, features = build_optimized_model(input_shape=(600, 600, 3))
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
    # Build and test
    optimized_model, features = build_optimized_model(input_shape=(600, 600, 3))
    input_tensor = tf.random.normal([1,600, 600,3])
    output = optimized_model(input_tensor)
    optimized_model.summary()
    # Define or load your model and input_tensor above this block
    measure_inference_time(optimized_model, input_tensor=input_tensor)

    # save_model()

    #################################### Convert Model to TFLite ######################################################
    # input_shape = (600, 600, 3)
    # keras_model, _ = build_optimized_model(input_shape=input_shape)
    # tflite_model_path = convert_to_tflite(keras_model)

    # run_tflite_inference(tflite_model_path)
