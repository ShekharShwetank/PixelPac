import cv2
import numpy as np
import tensorflow as tf
import time
import csv

# --- File Setup for Logging ---
LOG_FILE = '/home/ank/Desktop/CAT/CAT_TEST/Saved_tflite_models/depth_log.csv'
csv_file = open(LOG_FILE, 'w', newline='')
csv_writer = csv.writer(csv_file)
header = ['Timestamp'] + [f'Grid_{r+1}_{c+1}' for r in range(5) for c in range(5)]
csv_writer.writerow(header)
# --- End File Setup ---


# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="/home/ank/Desktop/CAT/CAT_TEST/Saved_tflite_models/adabins_quant_1.tflite")# change path as required
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ImageNet normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 160))
    img = img.astype(np.float32) / 255.0
    img = (img - IMG_MEAN) / IMG_STD
    img = np.expand_dims(img, axis=0)
    return img.astype(input_details[0]['dtype'])

def postprocess_depth(depth_map, upscale_shape=(900, 900)):
    depth_map_smoothed = cv2.GaussianBlur(depth_map, (5, 5), 0)
    depth_map_upscaled = cv2.resize(depth_map_smoothed, upscale_shape, interpolation=cv2.INTER_CUBIC)
    
    # =================================================================
    # === CRITICAL FIX: REVERSE THE NORMALIZATION FOR THE HEATMAP =====
    # =================================================================
    # Map the smallest depth value (min, closer) to 255 (red)
    # and the largest depth value (max, farther) to 0 (blue).
    depth_norm = cv2.normalize(depth_map_upscaled, None, 255, 0, cv2.NORM_MINMAX)
    # =================================================================

    depth_uint8 = depth_norm.astype(np.uint8)
    
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    return heatmap

def display_grid_values(image, true_depth_map):
    h, w = true_depth_map.shape
    display_h, display_w, _ = image.shape
    
    grid_rows, grid_cols = 5, 5
    cell_h, cell_w = h // grid_rows, w // grid_cols
    display_cell_h, display_cell_w = display_h // grid_rows, display_w // grid_cols

    avg_depth_values = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            depth_cell = true_depth_map[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            avg_depth = np.mean(depth_cell) if depth_cell.size > 0 else 0
            avg_depth_values.append(avg_depth)
            
            text = f"{avg_depth:.2f}"
            text_x = c * display_cell_w + (display_cell_w // 4)
            text_y = r * display_cell_h + (display_cell_h // 2)

            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (c * display_cell_w, r * display_cell_h), 
                          ((c + 1) * display_cell_w, (r + 1) * display_cell_h), (0, 0, 0), 1)

    return avg_depth_values

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting real-time TFLite depth estimation. Press 'q' to quit.")
print(f"Logging grid data to {LOG_FILE}")

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        input_img = preprocess_frame(frame)
        
        t1 = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_img)
        interpreter.invoke()
        
        # Get the raw disparity map from the model
        pred_disparity_map = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]
        
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-6 
        true_depth_map = 1.0 / (pred_disparity_map + epsilon)
        
        t2 = time.time()

        # Create the visual heatmap from the TRUE depth map
        heatmap = postprocess_depth(true_depth_map, upscale_shape=(900, 900))

        # Calculate, Display, and Log Grid Values using the TRUE depth map
        grid_values = display_grid_values(heatmap, true_depth_map)
        
        # Log the true depth data to the CSV file
        current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        csv_writer.writerow([current_timestamp] + grid_values)
        
        cv2.imshow('Depth Heatmap', heatmap)

        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        print(f"Inference time: {(t2 - t1):.3f} sec | Running FPS: {fps:.2f}", end='\r')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # --- Cleanup ---
    print(f"\nClosing resources and saving log file.")
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")