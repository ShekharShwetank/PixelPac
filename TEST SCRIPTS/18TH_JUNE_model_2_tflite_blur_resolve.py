import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="/home/ank/Desktop/CAT/Archive_2/model_best.tflite", num_threads=4)
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

def make_video_from_mp4(
    input_video_path,
    m, c,
    output_video_path='output.mp4',
    num_points_per_axis=6,
    fps=5):

    H, W = 256, 256
    out_width = 3 * W
    out_height = H

    # Prepare video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_video_path}")

    # Use input video FPS if available
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps > 0:
        fps = input_fps

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))
    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_video_path}")

    xs = np.linspace(0, W - 1, num_points_per_axis, dtype=int)
    ys = np.linspace(0, H - 1, num_points_per_axis, dtype=int)
    points = [(x, y) for y in ys for x in xs]

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
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

        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        frame_idx += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"Single combined video saved at: {output_video_path}")

m, c = (-0.34858393669128426, 11.856438636779785)
output_dir = "/home/ank/Desktop/CAT/Archive_2/output"
os.makedirs(output_dir, exist_ok=True)
make_video_from_mp4(
    input_video_path="/home/ank/Desktop/CAT/CAT_TEST/input.mp4",
    m=m,
    c=c,
    output_video_path=os.path.join(output_dir, "tflite_circle_depth_depth_gt_aligned_6_sorted_building.mp4"),
    num_points_per_axis=6
)