import cv2
import numpy as np
import tensorflow as tf
import time
from enhanced_depth_processor import EnhancedDepthProcessor

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="./Saved_tflite_models/adabins_quant_1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Normalization constants
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

# Enhanced processor initialization
processor = EnhancedDepthProcessor(
    temporal_buffer_size=3,
    bilateral_d=5,
    bilateral_sigma_color=50,
    bilateral_sigma_space=50,
    guided_radius=2,
    guided_eps=0.01
)

# Get model input size
input_h, input_w = input_details[0]['shape'][1:3]

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_w, input_h))
    img = img.astype(np.float32) / 255.0
    img = (img - IMG_MEAN) / IMG_STD
    img = np.expand_dims(img, axis=0)
    return img.astype(input_details[0]['dtype'])

def postprocess_depth_enhanced(depth_map, rgb_frame, processor, upscale_shape=(480, 480)):
    # Resize RGB to match depth for processing
    h, w = depth_map.shape
    rgb_resized = cv2.resize(rgb_frame, (w, h), interpolation=cv2.INTER_LINEAR)
    enhanced_depth, _, metrics = processor.process_frame(depth_map, rgb_resized)

    # Normalize depth to [0, 1]
    enhanced_depth = enhanced_depth.astype(np.float32)
    depth_norm = cv2.normalize(enhanced_depth, None, 0, 1, cv2.NORM_MINMAX)
    # Invert so that nearer = hot, farther = cool
    depth_norm = 1.0 - depth_norm

    # Upscale for display
    depth_up = cv2.resize(depth_norm, upscale_shape, interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = (255 * depth_up).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_PLASMA)

    # Also resize RGB for overlay
    rgb_up = cv2.resize(rgb_frame, upscale_shape, interpolation=cv2.INTER_LINEAR)

    # Overlay: alpha blend
    alpha = 0.5
    overlay = cv2.addWeighted(rgb_up, 1 - alpha, heatmap_color, alpha, 0)

    # Distance measurement points (you can adjust these)
    points = [
        (int(upscale_shape[0]*0.2), int(upscale_shape[1]*0.2)),
        (int(upscale_shape[0]*0.5), int(upscale_shape[1]*0.2)),
        (int(upscale_shape[0]*0.8), int(upscale_shape[1]*0.2)),
        (int(upscale_shape[0]*0.3), int(upscale_shape[1]*0.7)),
        (int(upscale_shape[0]*0.7), int(upscale_shape[1]*0.7)),
    ]

    # Estimate real-world distance (you may need to calibrate this scale!)
    min_depth, max_depth = 0.3, 3.0  # meters, adjust as needed
    for pt in points:
        x, y = pt
        # Use upscaled normalized inverted depth for distance calculation
        d_val = depth_up[y, x]  # value in [0,1]
        dist = min_depth + (max_depth - min_depth) * d_val
        cv2.circle(overlay, (x, y), 12, (0, 255, 0), 2)
        cv2.putText(overlay, f"{dist:.1f}m", (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return overlay, heatmap_color, metrics

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting enhanced real-time depth estimation. Press 'q' to quit.")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    input_img = preprocess_frame(frame)
    t1 = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    pred_depth = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]
    t2 = time.time()

    # Enhanced postprocessing: overlay and heatmap, both same size
    overlay, heatmap, metrics = postprocess_depth_enhanced(pred_depth, frame, processor)

    # Side-by-side display (optional)
    combined = np.hstack((overlay, heatmap))
    cv2.imshow('RGB+Depth Overlay | Depth Heatmap', combined)

    frame_count += 1
    fps = frame_count / (time.time() - start_time)
    print(f"Model: {(t2 - t1):.3f}s | Enhancement: {metrics['total_time']:.3f}s | FPS: {fps:.2f}", end='\r')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nAverage FPS: {fps:.2f}")
