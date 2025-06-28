import tensorflow as tf
from tensorflow.keras import layers, models
import time
import numpy as np
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)


# Depthwise separable conv block
def separable_conv_block(x, out_channels, strides=1):
    x = layers.SeparableConv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    return x

# Encoder block: downsample + 1 conv
def encoder_block(x, out_channels):
    x = separable_conv_block(x, out_channels, strides=2)  # Downsample
    x = separable_conv_block(x, out_channels, strides=1)
    return x

# Decoder block: upsample + concat + 2 convs
def upsample_block(x, skip, out_channels):
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.Concatenate()([x, skip])
    x = separable_conv_block(x, out_channels)
    x = separable_conv_block(x, out_channels)
    return x

def Custom_Encoder_Decoder(input_shape=(176, 608, 3)):
    output_channels=64
    inputs = layers.Input(shape=input_shape)

    # Encoder
    e1 = encoder_block(inputs, 32)   # 176x608 -> 88x304
    # Removing e2 and e3 to maintain final shape after decoder

    bottleneck = separable_conv_block(e1, 64, strides=1)  # Keep resolution 88x304

    # Decoder
    # No skip connections or upsampling required to match 88x304
    output = layers.Conv2D(output_channels, kernel_size=3, padding='same', activation='relu')(bottleneck)  # 88x304x64

    model = models.Model(inputs=inputs, outputs=output, name='FastDepthLite_Custom')
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
        print()
        times.append(end - start)

        inference_time = (end - start) * 1000  # milliseconds
        print(f"Inference output shape = {_.shape}, time = {inference_time:.2f} ms")

    avg_time = sum(times) / len(times)
    print(f"Average inference time over {num_runs} runs: {avg_time:.6f} seconds")


import os

def save_model():
    model = Custom_Encoder_Decoder()
    save_path = os.path.join(os.path.dirname(__file__), "model.h5")
    model.save(save_path)
    print(f"Model saved at: {save_path}")

    
# def convert_to_tflite(keras_model, tflite_model_path='encoder_decoder_draft1.tflite'):
#     # Create a converter object from the Keras model
#     converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
#     # Optional optimizations (comment out if not needed)
#     # converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     tflite_model = converter.convert()
    
#     # Save the converted model to disk
#     with open(tflite_model_path, 'wb') as f:
#         f.write(tflite_model)
#     print(f"TFLite model saved to: {tflite_model_path}")
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

# Test
if __name__ == "__main__":
    model = Custom_Encoder_Decoder()
    model.summary()

    dummy = tf.random.normal([1, 176, 608, 3])
    out = model(dummy)
    print("Output shape:", out.shape)

    measure_inference_time(model, dummy)
    print(f"tf version : {tf.__version__}")
    #################################### Save as .h5 model ######################################################
    # save_model()

    # #################################### Convert Model to TFLite ######################################################
    # input_shape = (176, 608, 3)
    # keras_model = Custom_Encoder_Decoder(input_shape=input_shape)
    # tflite_model_path = convert_to_tflite(keras_model)

    # #################################### run inferecnce on lite model ######################################################
    # run_tflite_inference(tflite_model_path)
    


    
