import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
import time

################################## CODE PARAMETER ####################################
video_path = "input_road.mp4"  # Replace with your input video path
output_video_path = "output_road.mp4"
width = 240*2
height = 240
model_path = "model.tflite"
image_shape = [240,240]

print(f"Starting the Model load process: ")
t1 = time.time()
# Load the TFLite model with 4 threads
interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=4)
interpreter.allocate_tensors()
print(f"Model loaded successfully in {(time.time() - t1)*1000:.2f} ms")

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("Error: Could not open video file")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height))

# Create display window
cv2.namedWindow('Depth Estimation', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Depth Estimation', width * 2, height)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    print(f"Processing frame {frame_count}")
    
    # Convert frame to model input
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb = tf.convert_to_tensor(img_rgb, dtype=tf.float32) / 255.0
    img_resized = tf.image.resize(img_rgb, image_shape, method='bilinear')
    input_data = tf.expand_dims(img_resized, axis=0)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    t = time.time()
    interpreter.invoke()
    print(f"Inference time: {(time.time() - t)*1000:.2f} ms")
    
    # Get the output tensor(s)
    outputs = {}
    for output in output_details:
        output_data = interpreter.get_tensor(output['index'])
        outputs[output['name']] = output_data
    
    # Process depth map
    depthmap = outputs["Identity_1"]
    depth_map = depthmap[0, 0, :, :]
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_vis = (255 * (depth_map - depth_min) / (depth_max - depth_min + 1e-6)).astype(np.uint8)
    depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
    # Resize for display
    depth_vis_color = cv2.resize(depth_vis_color, (width, height))
    original_vis = cv2.resize(frame, (width, height))
    
    # Create side-by-side view
    side_by_side = np.hstack((original_vis, depth_vis_color))
    
    # Display the frame in real-time
    cv2.imshow('Depth Estimation', side_by_side)
    
    # Write frame to video
    video_writer.write(side_by_side)
    
    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("Video processing completed!")