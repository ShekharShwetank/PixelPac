import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="/home/ank/Desktop/CAT/CAT_TEST/Saved_tflite_models/ADALITE_TFLITE.tflite", num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded.")

def preprocess_frame(frame, inputWidth=256, inputHeight=256):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (inputWidth, inputHeight), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    normalized_img = ((img_resized / 255.0 - mean) / std).astype(np.float32)
    return normalized_img, img_rgb  # normalized for model, RGB for display

def plot_depth_with_points_fixed(depth_map, points, title, cmap='plasma', H=256, W=256):
    dpi = 100
    figsize = (W / dpi, H / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(depth_map, cmap=cmap)
    for (x, y) in points:
        value = f"{depth_map[y, x]:.2f}"
        circle = plt.Circle((x, y), radius=4, color='white', fill=True, alpha=0.8, linewidth=0)
        ax.add_patch(circle)
        ax.text(x, y, value, color='black', fontsize=6, ha='center', va='center')
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout(pad=0)
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

def run_realtime_camera(
    m, c,
    num_points_per_axis=6):

    H, W = 256, 256
    out_width = 3 * W
    out_height = H

    xs = np.linspace(0, W - 1, num_points_per_axis, dtype=int)
    ys = np.linspace(0, H - 1, num_points_per_axis, dtype=int)
    points = [(x, y) for y in ys for x in xs]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame.")
            break

        x_norm, raw_img_rgb = preprocess_frame(frame, inputWidth=W, inputHeight=H)
        raw_img_rgb = cv2.resize(raw_img_rgb, (W, H))

        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x_norm, axis=0))
        interpreter.invoke()
        pred_depth = interpreter.get_tensor(output_details[0]['index'])
        pred_depth = np.squeeze(pred_depth, axis=0).squeeze(-1)
        if pred_depth.shape != (H, W):
            pred_depth = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_CUBIC)

        aligned_depth = m * pred_depth + c

        pred_vis = plot_depth_with_points_fixed(pred_depth, points, "Predicted Depth", cmap='plasma', H=H, W=W)
        aligned_vis = plot_depth_with_points_fixed(aligned_depth, points, "Aligned Depth", cmap='magma', H=H, W=W)

        pred_vis = pred_vis.astype(np.uint8)
        aligned_vis = aligned_vis.astype(np.uint8)
        raw_img_rgb = raw_img_rgb.astype(np.uint8)

        combined = np.hstack([
            raw_img_rgb,
            pred_vis,
            aligned_vis
        ])

        assert combined.shape == (out_height, out_width, 3), \
            f"Frame size mismatch: {combined.shape} vs {(out_height, out_width, 3)}"

        cv2.imshow("Original | Predicted Depth | Aligned Depth", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera stream ended.")

m, c = (-0.34858393669128426, 11.856438636779785)
run_realtime_camera(
    m=m,
    c=c,
    num_points_per_axis=6
)