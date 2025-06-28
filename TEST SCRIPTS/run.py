import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
import time



################################## CODE PARAMETER ####################################
image_dir = "test_data"
output_video_path = "output.mp4"
width = 240*2
height = 240
model_path = "model.tflite"
image_shape = [240,240]



def prerocess_input(img_path):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = tf.convert_to_tensor(img_rgb, dtype=tf.float32) / 255.0
    img_resized = tf.image.resize(img_rgb, image_shape, method='bilinear')
    input_img = tf.expand_dims(img_resized, axis=0)
    return input_img, img_bgr


print(f"Starting the Model load porcess : ")
t1 = time.time()
# Load the TFLite model with 4 threads
interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=4)
interpreter.allocate_tensors()
print(f"Model loaded successfully in {(time.time() - t1)*1000:.2f}  ms" )

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data
# Make sure input shape and dtype match the model's expected input
input_shape = input_details[0]['shape']  # e.g., [1, 240, 240, 3]
input_dtype = input_details[0]['dtype']

image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
], key=lambda x: int(os.path.splitext(x)[0]))

# Define video writer (side by side width = 2 * width)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (width * 2, height))

for file in image_files:
    img_path = os.path.join(image_dir,file )
    print(img_path)
    input_data, img_bgr = prerocess_input(img_path)
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    print(f"Started inference : ")
    t = time.time()
    # Run inference
    interpreter.invoke()
    print(f"Ended inference , time taken : {(time.time() - t)*1000:.2f} ms")

    # Get the output tensor(s)
    outputs = {}
    for output in output_details:
        output_data = interpreter.get_tensor(output['index'])
        outputs[output['name']] = output_data

    # dict_keys(['Identity', 'Identity_1'])


    depthmap = outputs["Identity_1"]
    depth_map = depthmap[0, 0, :, :]
    # print(depth_map.shape)
    # Normalize depth map to 0-255 for visualization
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_vis = (255 * (depth_map - depth_min) / (depth_max - depth_min + 1e-6)).astype(np.uint8)
    depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Resize depth map to original image height/width for side-by-side display
    depth_vis_color = cv2.resize(depth_vis_color, (width, height))
    original_vis = cv2.resize(img_bgr, (width, height))
    
    # Concatenate original and depth map side by side
    side_by_side = np.hstack((original_vis, depth_vis_color))

     # Write to video
    video_writer.write(side_by_side)

video_writer.release()

    
    

    


