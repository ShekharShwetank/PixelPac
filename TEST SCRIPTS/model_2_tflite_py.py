image_path = "/home/ank/Desktop/CAT/Archive_1/0.png"
import cv2
import numpy as np

import time


inputWidth, inputHeight = 256,256
    
img_path = image_path
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_original = img
img_original_resized = cv2.resize(img_original, (256, 256), interpolation=cv2.INTER_LINEAR)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image_height, image_width, image_channel = img.shape
input_image = cv2.resize(img, (inputWidth, inputHeight), interpolation=cv2.INTER_CUBIC).astype(np.float32)

# Scale input pixel values to -1 to 1
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


reshaped_img = input_image.reshape(1, inputHeight, inputWidth, 3)    
img_input = ((input_image/255.0 - mean) / std).astype(np.float32)

img_input = np.expand_dims(img_input, axis=0)

import tensorflow as tf
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="/home/ank/Desktop/CAT/Archive_1/model.tflite")

# Allocate tensors (must be done before inference)
interpreter.allocate_tensors()

# Get input and output tensor info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Example: Create a dummy input with the correct shape
# Note: input_details[0]['shape'] gives you the shape expected by the model.
input_shape = input_details[0]['shape']

for i in range(20):
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_input)
    t = time.time()
    # Run inference
    interpreter.invoke()
    print(f"time taken {(time.time() - t)*1000} ms")
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("Model output:", output_data.shape)

output_data = np.squeeze(output_data, axis=0)
