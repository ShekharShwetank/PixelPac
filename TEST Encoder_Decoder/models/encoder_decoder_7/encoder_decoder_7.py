import tensorflow as tf
from tensorflow.keras import layers, models

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# Depthwise separable conv block: ConvDW + ConvPoint + BN + Activation
def separable_conv_block(x, out_channels, strides=1):
    x = layers.SeparableConv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)  # ReLU6 for edge device compatibility
    return x

# Encoder block: downsample with stride=2 separable conv, then 1-2 conv blocks
def encoder_block(x, out_channels):
    x = separable_conv_block(x, out_channels, strides=2)  # downsample spatial size by 2
    x = separable_conv_block(x, out_channels, strides=1)
    return x

# Decoder upsample block: bilinear upsample + concat + separable conv block(s)
def upsample_block(x, skip, out_channels):
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.Concatenate()([x, skip])
    x = separable_conv_block(x, out_channels)
    x = separable_conv_block(x, out_channels)
    return x

def FastDepthLite(input_shape=(480, 480, 3), output_channels=64):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    e1 = encoder_block(inputs, 32)   # 480 -> 240
    e2 = encoder_block(e1, 64)       # 240 -> 120
    e3 = encoder_block(e2, 128)      # 120 -> 60

    bottleneck = separable_conv_block(e3, 256, strides=1)  # 60x60

    # Decoder with correct skip connections
    d1 = upsample_block(bottleneck, e2, 128)  # 60->120 concat with e2 (120x120)
    d2 = upsample_block(d1, e1, 64)           # 120->240 concat with e1 (240x240)

    output = layers.Conv2D(output_channels, kernel_size=3, padding='same', activation='relu')(d2)  # 240x240x64

    model = models.Model(inputs=inputs, outputs=output, name='FastDepthLite')
    return model

    return model
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
    model = FastDepthLite()
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
    

# Test summary
if __name__ == "__main__":
    model = FastDepthLite()
    model.summary()

    # Test dummy input
    dummy = tf.random.normal([1, 480, 480, 3])
    out = model(dummy)
    print("Output shape:", out.shape)  # should be (1, 240, 240, 64)

    # Define or load your model and input_tensor above this block
    measure_inference_time(model, dummy)

    # # save_model()

    # ################################### Convert Model to TFLite ######################################################
    # input_shape = (480, 480, 3)
    # keras_model = FastDepthLite(input_shape=input_shape)
    # tflite_model_path = convert_to_tflite(keras_model)

    # run_tflite_inference(tflite_model_path)