import cv2
import numpy as np
import tensorflow as tf
import time
import csv
import signal

# --- File Setup for Logging ---
LOG_FILE = './depth_log.csv'
csv_file = open(LOG_FILE, 'w', newline='')
csv_writer = csv.writer(csv_file)
header = ['Timestamp'] + [f'Grid_{r+1}_{c+1}' for r in range(25) for c in range(25)]
csv_writer.writerow(header)
# --- End File Setup ---

# --- Model Setup ---
MODEL_PATH = "/home/ank/Desktop/CAT/CAT_TEST/Saved_tflite_models/adabins_quant_1.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
    depth_map_upscaled = cv2.resize(depth_map, upscale_shape, interpolation=cv2.INTER_CUBIC)
    depth_norm = cv2.normalize(depth_map_upscaled, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    return heatmap

def display_grid_values(image, raw_depth_map):
    h, w = raw_depth_map.shape
    display_h, display_w, _ = image.shape
    grid_rows, grid_cols = 25, 25
    cell_h, cell_w = h // grid_rows, w // grid_cols
    display_cell_h, display_cell_w = display_h // grid_rows, display_w // grid_cols
    avg_depth_values = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            start_row, end_row = r * cell_h, (r + 1) * cell_h
            start_col, end_col = c * cell_w, (c + 1) * cell_w
            depth_cell = raw_depth_map[start_row:end_row, start_col:end_col]
            avg_depth = np.mean(depth_cell) if depth_cell.size > 0 else 0
            avg_depth_values.append(avg_depth)
            text = f"{1.0/avg_depth:.2f}" if avg_depth != 0 else "inf"
            text_x = c * display_cell_w + (display_cell_w // 4)
            text_y = r * display_cell_h + (display_cell_h // 2)
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(image, (c * display_cell_w, r * display_cell_h),
                          ((c + 1) * display_cell_w, (r + 1) * display_cell_h), (0, 0, 0), 1)
    return avg_depth_values

running = True
def cleanup_handler(signum=None, frame=None):
    global running
    if running:
        print("\nShutdown signal received. Exiting loop...")
        running = False

def main():
    global running
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started.")
    frame_count = 0
    start_time = time.time()

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            input_img = preprocess_frame(frame)
            t1 = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_img)
            interpreter.invoke()
            pred_depth_raw = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]
            t2 = time.time()
            heatmap = postprocess_depth(pred_depth_raw, upscale_shape=(900, 900))
            grid_values = display_grid_values(heatmap, pred_depth_raw)
            current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            csv_writer.writerow([current_timestamp] + grid_values)
            frame_resized = cv2.resize(frame, (heatmap.shape[1], heatmap.shape[0]))
            combined = np.hstack((frame_resized, heatmap))
            cv2.imshow("Original | Depth Heatmap", combined)
            frame_count += 1
            fps = frame_count / (time.time() - start_time)
            print(f"Inference: {(t2 - t1):.3f}s | FPS: {fps:.2f}", end='\r')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        print("\nClosing resources and saving log file.")
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")

if __name__ == '__main__':
    main()