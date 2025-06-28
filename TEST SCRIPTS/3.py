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

def postprocess_depth(depth_map, upscale_shape=(640, 480)):
    # 1. Apply a Gaussian blur to the raw, low-resolution depth map to smooth it.
    depth_map_smoothed = cv2.GaussianBlur(depth_map, (5, 5), 0)
    
    # 2. Upscale the smoothed depth map for better display.
    depth_map_upscaled = cv2.resize(depth_map_smoothed, upscale_shape, interpolation=cv2.INTER_CUBIC)
    
    # 3. Normalize the upscaled map to the 0-255 range to apply a colormap.
    depth_norm = cv2.normalize(depth_map_upscaled, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    
    # 4. Apply a colormap.
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    return heatmap

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting real-time TFLite depth estimation. Press 'q' to quit.")

frame_count = 0
start_time = time.time()

# Get the original frame dimensions from the camera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
display_height = 480  # Set a fixed height for the side-by-side display
display_width = int(frame_width * (display_height / frame_height))


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # --- MODEL INFERENCE ---
    input_img = preprocess_frame(frame)
    t1 = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    # The raw output from the model before any post-processing
    pred_depth_raw = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]
    print(pred_depth_raw)
    t2 = time.time()

    # --- VISUALIZATION PREPARATION ---

    # 1. Post-process the raw depth for heatmap visualization
    heatmap = postprocess_depth(pred_depth_raw, upscale_shape=(display_width, display_height))
    
    # 2. Resize original frame to match the heatmap's display height for side-by-side view
    input_frame_display = cv2.resize(frame, (display_width, display_height))

    # --- NEW: DISPLAY RAW DEPTH VALUE ---
    # Get the physical depth value from the center of the raw prediction
    h, w = pred_depth_raw.shape
    center_x, center_y = w // 2, h // 2
    center_depth_value = pred_depth_raw[center_y, center_x]
    
    # Add a visual marker (a circle) to the center of the heatmap
    marker_pos = (display_width // 2, display_height // 2)
    cv2.circle(heatmap, marker_pos, 5, (255, 255, 255), 2, cv2.LINE_AA) # White circle

    # Prepare the text to display
    depth_text = f"Depth at center: {center_depth_value:.2f} m" # Assuming model outputs meters
    
    # Overlay the text onto the heatmap
    cv2.putText(heatmap, depth_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # --- NEW: SIDE-BY-SIDE DISPLAY ---
    # Concatenate the input frame and the heatmap horizontally
    combined_display = np.hstack((input_frame_display, heatmap))
    
    # Show the combined view in a single window
    cv2.imshow('Real-time Monocular Depth Estimation', combined_display)

    # --- FPS CALCULATION AND EXIT LOGIC ---
    frame_count += 1
    fps = frame_count / (time.time() - start_time)
    print(f"Inference time: {(t2 - t1):.3f} sec | Running FPS: {fps:.2f}", end='\r')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()