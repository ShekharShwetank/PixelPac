import os
import cv2
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def preprocess_image(img_path, inputWidth=256, inputHeight=256):
    """Preprocess input image for the model."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (inputWidth, inputHeight), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    normalized_img = ((img_resized / 255.0 - mean) / std).astype(np.float32)
    return normalized_img, img_rgb  # normalized for model, RGB for display

def plot_depth_with_points_fixed(depth_map, points, title, cmap='plasma', H=256, W=256):
    """
    Return an RGB image of the depth map with:
    ✅ High DPI
    ✅ Clear circle + text placed beside each point
    ✅ Readable font
    """
    depth_map = np.squeeze(depth_map)

    # High DPI for crisp output
    dpi = 300
    figsize = (W / dpi, H / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(depth_map, cmap=cmap)

    for (x, y) in points:
        value = f"{depth_map[y, x]:.2f}"

        # Circle
        circle = plt.Circle((x, y), radius=2, color='red', fill=True, antialiased=True)
        ax.add_patch(circle)

        # Text beside the circle
        ax.text(x + 5, y - 5, value,
                color='white',
                fontsize=2,
                fontweight='bold',
                family='DejaVu Sans',
                ha='left',
                va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.0))

    ax.set_title(title, fontsize=6)
    ax.axis('off')
    fig.tight_layout(pad=0)

    canvas = FigureCanvas(fig)
    canvas.draw()

    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

# if __name__ == "__main__":
#     # Linear regression coefficients
#     # M, C = (-0.48565328, 15.496328)
#     M, C = (-2.3612990379333496, 12.253952026367188)
if __name__ == "__main__":
    # Linear regression coefficients
    # M, C = (-2.3612990379333496, 12.253952026367188)
    # Global m: -0.3485839366912842, Global c: 11.856438636779785
    M, C = (-0.34858393669128426,11.856438636779785)
    

    # Load TFLite model
    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path="/home/ank/Desktop/CAT/Archive_2/model_best.tflite", num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded.")

    # Correct paths
    img_dir = "data/img"  # a directory, not a single image
    fps = 10
    H, W = 256, 256

    # Grid points
    num_points_per_axis = 6
    xs = np.linspace(0, W - 1, num_points_per_axis, dtype=int)
    ys = np.linspace(0, H - 1, num_points_per_axis, dtype=int)
    points = [(x, y) for y in ys for x in xs]

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # fallback to 'mp4v' if needed
    video_writer = cv2.VideoWriter(
        'aligned_output.mp4',
        fourcc,
        fps,
        (2 * W, H)
    )

    if not video_writer.isOpened():
        raise RuntimeError("Failed to open video writer. Check codec or filename.")

    # Get sorted image list
    image_list = sorted(os.listdir(img_dir), key=lambda x: int(os.path.splitext(x)[0]))

    for i, image_name in enumerate(image_list):
        img_path = os.path.join(img_dir, image_name)
        print(f"Processing: {img_path}")

        # Preprocess
        normalized_img, raw_img_rgb = preprocess_image(img_path)
        raw_img_rgb = cv2.resize(raw_img_rgb, (W, H), interpolation=cv2.INTER_CUBIC)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(normalized_img, axis=0))
        t = time.time()
        interpreter.invoke()
        print(f"Inference time: {(time.time() - t)*1000:.2f} ms")

        # Get & align depth
        output_depth_map = interpreter.get_tensor(output_details[0]['index'])
        depth_map_2d = np.squeeze(output_depth_map)
        aligned_depth_map = depth_map_2d * M + C

        # Plot depth with points
        aligned_vis = plot_depth_with_points_fixed(
            aligned_depth_map, points, title="Depth Map", cmap='magma', H=H, W=W
        )

        # Combine original and depth side by side
        combined = np.hstack([
            raw_img_rgb,
            cv2.resize(aligned_vis, (W, H), interpolation=cv2.INTER_CUBIC)
        ])

        # Write to video (OpenCV expects BGR)
        video_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print("✅ High-quality video saved: aligned_output.mp4")