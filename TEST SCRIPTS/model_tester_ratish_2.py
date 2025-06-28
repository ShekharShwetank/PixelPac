import cv2
import numpy as np
import tensorflow as tf
import time

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="/home/ank/Desktop/CAT/CAT_TEST/Saved_tflite_models/adabins_quant_1.tflite")# change path as required
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ImageNet normalization (as in your training)
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 160))  # (width, height)
    img = img.astype(np.float32) / 255.0
    img = (img - IMG_MEAN) / IMG_STD
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img.astype(input_details[0]['dtype'])

def postprocess_depth(depth_map, upscale_shape=(1250, 1250)):
    # 1. Apply a Gaussian blur to the raw, low-resolution depth map to smooth it.
    # The kernel size (e.g., (5, 5)) can be tuned. A larger kernel means more blur.
    depth_map_smoothed = cv2.GaussianBlur(depth_map, (5, 5), 0)

    # 2. Upscale the smoothed depth map for better display.
    # cv2.INTER_CUBIC is a good choice for upscaling.
    depth_map_upscaled = cv2.resize(depth_map_smoothed, upscale_shape, interpolation=cv2.INTER_CUBIC)

    # 3. Normalize the upscaled map to the 0-255 range to apply a colormap.
    depth_norm = cv2.normalize(depth_map_upscaled, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)

    # 4. Apply a colormap. Try different ones to see what works best for you.
    # Other options: cv2.COLORMAP_JET, cv2.COLORMAP_PLASMA, cv2.COLORMAP_MAGMA
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    return heatmap

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting real-time TFLite depth estimation. Press 'q' to quit.")

frame_count = 0
start_time = time.time()

# --- Main Loop ---
# (Everything before the 'while True:' loop remains the same)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    input_img = preprocess_frame(frame)
    t1 = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    pred_depth = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]  # This is your raw depth map
    t2 = time.time()

    # --- NEW SECTION: Get Raw Depth Values ---
    
    # Get the dimensions of the raw depth map
    h, w = pred_depth.shape
    
    # Get the depth value at the center of the screen
    center_x, center_y = w // 2, h // 2
    center_depth = pred_depth[center_y, center_x]
    
    # Get the minimum and maximum depth values across the entire frame
    min_depth = np.min(pred_depth)
    max_depth = np.max(pred_depth)

    # --- END NEW SECTION ---

    # Post-process the depth map to create the visual heatmap
    heatmap = postprocess_depth(pred_depth, upscale_shape=(1250, 1250))
    
    # --- NEW SECTION: Display Raw Depth Values on the Heatmap ---
    
    # Format the text strings to display (assuming distance is in meters)
    center_text = f"Center Depth: {center_depth:.2f} m"
    min_max_text = f"Min: {min_depth:.2f} m | Max: {max_depth:.2f} m"

    # Put the text on the heatmap image.
    # cv2.putText(image, text, origin, font, scale, color, thickness)
    cv2.putText(heatmap, center_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(heatmap, min_max_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # --- END NEW SECTION ---

    cv2.imshow('Depth Heatmap', heatmap)

    frame_count += 1
    fps = frame_count / (time.time() - start_time)
    print(f"Inference time: {(t2 - t1):.3f} sec | Running FPS: {fps:.2f}", end='\r')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# (The cleanup code after the loop remains the same)

cap.release()
cv2.destroyAllWindows()
print(f"\nAverage FPS: {fps:.2f}")
