import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Lambda
import time
import os
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)


def FastDepth(input_shape=(300, 300, 3), output_channels=64):
    inputs = tf.keras.Input(shape=input_shape)

    # MobileNetV2 backbone with input tensor
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights='imagenet'
    )

    # Encoder layers for skip connections
    encoder_output = base_model.get_layer('block_13_expand_relu').output  # ~[10x10x576] for 300x300 input
    skip_1 = base_model.get_layer('block_6_expand_relu').output           # ~[19x19x192]
    skip_2 = base_model.get_layer('block_3_expand_relu').output           # ~[38x38x96]
    skip_3 = base_model.get_layer('block_1_expand_relu').output           # ~[75x75x24]

    # Decoder block 1: upsample encoder_output to skip_1 spatial size (~19x19)
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(encoder_output)  # [~20x20x256]
    x = layers.Concatenate()([x, skip_1])
    x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 1, padding='same', activation='relu')(x)

    # Decoder block 2: upsample to skip_2 spatial size (~38x38)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)  # [~38x38x128]
    x = layers.Lambda(lambda t: tf.image.resize(t, [75, 75]))(x)
    # print("shape is : ",x.shape)
    x = layers.Concatenate()([x, skip_2])
    x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)

    # Decoder block 3: upsample to skip_3 spatial size (~75x75)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)  # [~75x75x64]
    print(f"shape of x is : {x.shape}")
    print(f"shape of skip_3 is : {skip_3.shape}")
    # x = layers.Lambda(lambda t: tf.image.resize(t, tf.shape(skip_3)[1:3], method='nearest'))(x)
    x = layers.Concatenate()([x, skip_3])
    x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)

    # Final upsample to original input size (300x300)
    x = layers.Conv2DTranspose(64, 3, strides=4, padding='same')(x)  # upsample 75x75 -> 300x300
    x = layers.Conv2D(output_channels, 1, padding='same', activation='relu')(x)

    return models.Model(inputs, x, name="FastDepth_Tiny_300")


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
def run_tflite_inference(tflite_model_path, input_shape=(1, 300, 300, 3), num_runs=10):
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
    model = FastDepth(input_shape=(300, 300, 3), output_channels=64)
    model.summary()

    # Test with a dummy input
    dummy_input = tf.random.normal([1, 300, 300, 3])
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (1, 600, 600, 64)
    # Define or load your model and input_tensor above this block
    measure_inference_time(model, dummy_input)

    # save_model()


    ################################### Convert Model to TFLite ######################################################
    # input_shape = (300, 300, 3)
    # keras_model = FastDepth(input_shape=input_shape)
    # tflite_model_path = convert_to_tflite(keras_model)

    # run_tflite_inference(tflite_model_path)