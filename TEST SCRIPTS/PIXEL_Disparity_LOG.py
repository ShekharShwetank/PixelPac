#THE CODE THAT WORKS

import cv2
import numpy as np
import tensorflow as tf
import time
import csv

# --- File Setup for Logging ---
LOG_FILE = '/home/ank/Desktop/CAT/CAT_TEST/Saved_tflite_models/depth_log.csv'
# Create and prepare the CSV file for logging grid depth values
csv_file = open(LOG_FILE, 'w', newline='')
csv_writer = csv.writer(csv_file)
# Create header row for the CSV file (Timestamp + 25 grid cells)
header = ['Timestamp'] + [f'Grid_{r+1}_{c+1}' for r in range(25) for c in range(25)]
csv_writer.writerow(header)
# --- End File Setup ---


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

'''
# Original postprocess_depth function with Gaussian blur and colormap
def postprocess_depth(depth_map, upscale_shape=(1250, 1250)):
    # Apply a Gaussian blur to the raw, low-resolution depth map to smooth it.
    depth_map_smoothed = cv2.GaussianBlur(depth_map, (5, 5), 0)

    # Upscale the smoothed depth map for better display.
    depth_map_upscaled = cv2.resize(depth_map_smoothed, upscale_shape, interpolation=cv2.INTER_CUBIC)

    # Normalize the upscaled map to the 0-255 range to apply a colormap.
    depth_norm = cv2.normalize(depth_map_upscaled, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)

    # Apply a colormap.
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_RAINBOW)
    
    return heatmap
'''

def postprocess_depth(depth_map, upscale_shape=(900, 900)):
    # No Gaussian blur, just resize and colormap
    depth_map_upscaled = cv2.resize(depth_map, upscale_shape, interpolation=cv2.INTER_CUBIC)
    depth_norm = cv2.normalize(depth_map_upscaled, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    return heatmap

def display_grid_values(image, raw_depth_map):
    """
    Divides the screen into a 5x5 grid, calculates the average raw depth for each cell,
    and displays the value on the image.
    Returns the list of calculated average values.
    """
    h, w = raw_depth_map.shape
    display_h, display_w, _ = image.shape
    
    grid_rows, grid_cols = 25, 25
    cell_h, cell_w = h // grid_rows, w // grid_cols
    display_cell_h, display_cell_w = display_h // grid_rows, display_w // grid_cols

    avg_depth_values = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            # Define the region of interest in the raw depth map
            start_row, end_row = r * cell_h, (r + 1) * cell_h
            start_col, end_col = c * cell_w, (c + 1) * cell_w
            
            # Extract the cell from the raw depth map
            depth_cell = raw_depth_map[start_row:end_row, start_col:end_col]
            
            # Calculate the average depth, handling cases with no depth info
            avg_depth = np.mean(depth_cell) if depth_cell.size > 0 else 0
            avg_depth_values.append(avg_depth)
            
            # Prepare text to display
            text = f"{1.0/avg_depth:.2f}"
            
            # Calculate text position on the display image
            text_x = c * display_cell_w + (display_cell_w // 4)
            text_y = r * display_cell_h + (display_cell_h // 2)

            # Draw the text on the image
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw grid lines for clarity
            cv2.rectangle(image, (c * display_cell_w, r * display_cell_h), 
                          ((c + 1) * display_cell_w, (r + 1) * display_cell_h), (0, 0, 0), 1)

    return avg_depth_values

USE_VIDEO = True # Set to False to use webcam

if USE_VIDEO:
    cap = cv2.VideoCapture('/home/ank/Desktop/CAT/CAT_TEST/input_road.mp4')
else:
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
        # Get the raw depth map (this contains the actual distance values)
        pred_depth_raw = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]
        t2 = time.time()

        # Create the visual heatmap from the raw depth map
        heatmap = postprocess_depth(pred_depth_raw, upscale_shape=(900, 900))
        
        # --- New Section: Calculate, Display, and Log Grid Values ---
        # This function modifies the 'heatmap' image in place and returns the grid values
        grid_values = display_grid_values(heatmap, pred_depth_raw)
        
        # Log the data to the CSV file
        current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        csv_writer.writerow([current_timestamp] + grid_values)
        # --- End New Section ---

# ...existing code...

        # Resize original frame to match heatmap size for side-by-side display
        frame_resized = cv2.resize(frame, (heatmap.shape[1], heatmap.shape[0]))
        combined = np.hstack((frame_resized, heatmap))

        cv2.imshow('Original | Depth Heatmap', combined)

        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        print(f"Inference time: {(t2 - t1):.3f} sec | Running FPS: {fps:.2f}", end='\r')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# ...existing code...

        #cv2.imshow('Depth Heatmap', heatmap)

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


