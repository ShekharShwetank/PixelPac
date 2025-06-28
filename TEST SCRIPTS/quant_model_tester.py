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

def postprocess_depth(depth_map, upscale_shape=(320, 160)):
    # Upscale for display clarity
    depth_map = cv2.resize(depth_map, upscale_shape, interpolation=cv2.INTER_CUBIC)
    # Normalize for colormap
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    return heatmap

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting real-time TFLite depth estimation. Press 'q' to quit.")

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
    pred_depth = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]  # Remove batch and channel dims
    t2 = time.time()

    # Upscale and display heatmap for clarity
    heatmap = postprocess_depth(pred_depth, upscale_shape=(900, 900))
    cv2.imshow('Depth Heatmap', heatmap)

    frame_count += 1
    fps = frame_count / (time.time() - start_time)
    print(f"Inference time: {(t2 - t1):.3f} sec | Running FPS: {fps:.2f}", end='\r')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nAverage FPS: {fps:.2f}")
