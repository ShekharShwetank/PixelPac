import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Load TFLite model
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="/home/ank/Desktop/CAT/Archive_2/model_best.tflite", num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded.")


def preprocess_image(img_path, inputWidth=256, inputHeight=256):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (inputWidth, inputHeight), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    normalized_img = ((img_resized / 255.0 - mean) / std).astype(np.float32)
    return normalized_img, img  # normalized RGB for model, original BGR

def load_velodyne_depth(gt_path, output_size=(256, 256)):
    raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    raw_meters = raw.astype(np.float32) / 256.0  # for KITTI
    resized = cv2.resize(raw_meters, output_size, interpolation=cv2.INTER_NEAREST)
    return resized



def plot_depth_with_points_fixed(depth_map, points, title, cmap='plasma', H=256, W=256):
    """Plot depth map with circles + value text at given points."""
    dpi = 100
    figsize = (W / dpi, H / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(depth_map, cmap=cmap)

    for (x, y) in points:
        value = f"{depth_map[y, x]:.2f}"
        # Draw a small circle
        circle = plt.Circle((x, y), radius=4, color='white', fill=True, alpha=0.8, linewidth=0)
        ax.add_patch(circle)
        # Add text over it
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


def make_video(image_dir, gt_dir, model, m, c,
               output_video_path='output.mp4',
               num_points_per_axis=15,
               fps=5):

    # Robust numeric sort: expects filenames like '1.png', '2.png', ...
    image_list = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
    gt_list = sorted(os.listdir(gt_dir), key=lambda x: int(os.path.splitext(x)[0]))
    assert len(image_list) == len(gt_list), "Image and GT count mismatch."

    H, W = 256, 256
    out_width = 2 * W  # 2 columns
    out_height = 2 * H  # 2 rows

    # Safe codec for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_video_path}")

    xs = np.linspace(0, W - 1, num_points_per_axis, dtype=int)
    ys = np.linspace(0, H - 1, num_points_per_axis, dtype=int)
    points = [(x, y) for y in ys for x in xs]

    for img_name, gt_name in tqdm(zip(image_list, gt_list), total=len(image_list)):
        # --- Load & preprocess
        x_norm, img_bgr = preprocess_image(os.path.join(image_dir, img_name))
        img_bgr = cv2.resize(img_bgr, (W, H))

        velodyne = load_velodyne_depth(os.path.join(gt_dir, gt_name), output_size=(W, H))

        pred_depth = model(np.expand_dims(x_norm, axis=0)).numpy()
        pred_depth = np.squeeze(pred_depth, axis=0).squeeze(-1)

        if pred_depth.shape != (H, W):
            pred_depth = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_CUBIC)

        aligned_depth = m * pred_depth + c

        # --- Visualizations
        rgb_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pred_vis = plot_depth_with_points_fixed(pred_depth, points, "Predicted Depth", cmap='plasma', H=H, W=W)
        velo_vis = plot_depth_with_points_fixed(velodyne, points, "Velodyne GT", cmap='viridis', H=H, W=W)
        aligned_vis = plot_depth_with_points_fixed(aligned_depth, points, "Aligned Depth", cmap='magma', H=H, W=W)

        # --- Ensure shape & dtype
        pred_vis = cv2.resize(pred_vis, (W, H)).astype(np.uint8)
        velo_vis = cv2.resize(velo_vis, (W, H)).astype(np.uint8)
        aligned_vis = cv2.resize(aligned_vis, (W, H)).astype(np.uint8)
        rgb_frame = rgb_frame.astype(np.uint8)

        # --- Top row: [RGB | Pred]
        top_row = np.hstack([rgb_frame, pred_vis])

        # --- Bottom row: [Velodyne | Aligned]
        bottom_row = np.hstack([velo_vis, aligned_vis])

        # --- Final 2x2 grid
        combined = np.vstack([top_row, bottom_row])

        assert combined.shape == (out_height, out_width, 3), \
            f"Shape mismatch: got {combined.shape}, expected {(out_height, out_width, 3)}"

        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"2x2 grid video saved at: {output_video_path}")


def make_video_model_out_aligned_side_by_side(
    image_dir, gt_dir, m, c,
    output_video_path='combined_video.mp4',
    num_points_per_axis=15, fps=5):

    # Robust numeric sort
    image_list = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
    gt_list = sorted(os.listdir(gt_dir), key=lambda x: int(os.path.splitext(x)[0]))
    assert len(image_list) == len(gt_list), "Image and GT count mismatch."

    H, W = 256, 256
    out_width = 3 * W
    out_height = H

    # SAFER cross-platform MP4 codec: use H.264 or fallback to 'MJPG'
    # H.264 works well if you have FFMPEG backend installed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'avc1' or 'H264' if installed

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed to open. Check codec or file path: {output_video_path}")

    xs = np.linspace(0, W - 1, num_points_per_axis, dtype=int)
    ys = np.linspace(0, H - 1, num_points_per_axis, dtype=int)
    points = [(x, y) for y in ys for x in xs]

    for img_name, gt_name in tqdm(zip(image_list, gt_list), total=len(image_list)):
        # --- Load & preprocess
        x_norm, raw_img_bgr = preprocess_image(os.path.join(image_dir, img_name))
        raw_img_bgr = cv2.resize(raw_img_bgr, (W, H))

        velodyne = load_velodyne_depth(os.path.join(gt_dir, gt_name), output_size=(W, H))

        # pred_depth = model(np.expand_dims(x_norm, axis=0)).numpy()
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x_norm, axis=0))
        interpreter.invoke()
        pred_depth = interpreter.get_tensor(output_details[0]['index'])
        pred_depth = np.squeeze(pred_depth, axis=0).squeeze(-1)
        if pred_depth.shape != (H, W):
            pred_depth = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_CUBIC)

        aligned_depth = m * pred_depth + c

        pred_vis = plot_depth_with_points_fixed(pred_depth, points, "Predicted Depth", cmap='plasma', H=H, W=W)
        aligned_vis = plot_depth_with_points_fixed(aligned_depth, points, "Aligned Depth", cmap='magma', H=H, W=W)

        # --- Ensure all frames are uint8, RGB, and correct size
        raw_img_rgb = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB)  # your raw is BGR, so fix to RGB

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

        # Write as BGR
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Single combined video saved at: {output_video_path}")

m, c = (-0.34858393669128426,11.856438636779785)
make_video_model_out_aligned_side_by_side(
    image_dir="data/img",
    gt_dir="data/gt",
    m=m,
    c=c,
    output_video_path="/home/ank/Desktop/CAT/Archive_2/output/tflite_circle_depth_depth_gt_aligned_6_sorted.mp4",
    num_points_per_axis=6
)