import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import time
import os
import sys
import signal
import mmap
import fcntl

# Import Picamera2 for camera module 3
from picamera2 import Picamera2

# Ensure matplotlib is used in headless mode and then import necessary components
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plot generation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ======================================================================================
# FrameBuffer Display Class for Console-based Real-Time Output
# ======================================================================================
class FrameBuffer:
    """A class to handle displaying NumPy arrays on the Linux framebuffer."""
    def __init__(self, device='/dev/fb0'):
        self.fb_dev = os.open(device, os.O_RDWR)
        # FBIOGET_VSCREENINFO ioctl for variable screen info
        var_info_bytes = fcntl.ioctl(self.fb_dev, 0x4600, b'\0' * 160)
        self.width = int.from_bytes(var_info_bytes[0:4], 'little')
        self.height = int.from_bytes(var_info_bytes[4:8], 'little')
        self.bpp = int.from_bytes(var_info_bytes[24:28], 'little')
        self.screen_size = self.width * self.height * (self.bpp // 8)
        self.fb_map = mmap.mmap(self.fb_dev, self.screen_size, mmap.MAP_SHARED, mmap.PROT_WRITE)
        print(f"Framebuffer initialized: {self.width}x{self.height} @ {self.bpp}bpp")

    def draw(self, frame_rgb):
        """Draw a single RGB frame to the framebuffer."""
        # Resize frame to framebuffer resolution
        resized_frame = cv2.resize(frame_rgb, (self.width, self.height))
        
        # Convert to appropriate format for framebuffer BGRX or BGR565
        if self.bpp == 32:
            # Framebuffer expects BGRA, but our input is RGB. Convert RGB to BGRA.
            output_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGRA)
        elif self.bpp == 16:
            # Framebuffer expects BGR565. Convert RGB to BGR565.
            output_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR565)
        else:
            raise ValueError(f"Unsupported bits-per-pixel: {self.bpp}. Supported: 16, 32.")
            
        self.fb_map.seek(0)
        self.fb_map.write(output_frame.tobytes())

    def close(self):
        """Unmap memory and close the framebuffer device."""
        if hasattr(self, 'fb_map'): self.fb_map.close()
        if hasattr(self, 'fb_dev'): os.close(self.fb_dev)
        print("Framebuffer closed.")

# --- Global Objects & State for Graceful Shutdown ---
picam2, interpreter, display = None, None, None
running = True

def cleanup_handler(signum=None, frame=None):
    """Signal handler to initiate a clean shutdown."""
    global running
    if running:
        print("\nShutdown signal received. Exiting loop...")
        running = False

# --- Configuration ---
# Model path is absolute as provided in the original script
MODEL_PATH = "ADALITE_TFLITE.tflite" 
# Model input dimensions (H, W)
MODEL_INPUT_HEIGHT = 256
MODEL_INPUT_WIDTH = 256
# Display dimensions for each individual visualization (original, predicted, aligned)
# Each will occupy DISPLAY_WIDTH x DISPLAY_HEIGHT area
DISPLAY_WIDTH_SINGLE = 256
DISPLAY_HEIGHT_SINGLE = 256


# ======================================================================================
# Original Depth Estimation Functions (adapted for console environment)
# ======================================================================================

def preprocess_frame(frame, inputWidth, inputHeight):
    # The Picamera2 capture_array provides BGR. Convert to RGB for the model input.
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (inputWidth, inputHeight), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    normalized_img = ((img_resized / 255.0 - mean) / std).astype(np.float32)
    return normalized_img, img_rgb  # normalized for model, RGB for display

def plot_depth_with_points_fixed(depth_map, points, title, cmap='plasma', H=256, W=256):
    # This function uses matplotlib to create an image, then converts it to a numpy array.
    # We need to ensure the output is RGB for the framebuffer.
    # The original function already outputs RGB, so we mostly keep it.
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
    
    # Render the matplotlib figure to a numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig) # Close the figure to free up memory
    
    # Convert RGBA to RGB as required for our display pipeline
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

def main():
    global picam2, interpreter, display, running
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    #m, c = (-0.34858393669128426, 11.856438636779785) # Constants from your original script
    m, c = (-0.37965089082717896, 14.945058822631836)
    num_points_per_axis = 6

    # Calculate points for depth map visualization
    xs = np.linspace(0, MODEL_INPUT_WIDTH - 1, num_points_per_axis, dtype=int)
    ys = np.linspace(0, MODEL_INPUT_HEIGHT - 1, num_points_per_axis, dtype=int)
    points = [(x, y) for y in ys for x in xs]

    try:
        # Initialize FrameBuffer for display output
        display = FrameBuffer()

        print("Loading TFLite model...")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Model loaded.")

        # Initialize Picamera2
        picam2 = Picamera2()
        # Request BGR888 format from the camera to align with OpenCV's native format.
        # This reduces color conversion overhead.
        # Configure camera to output at a size that is easily cropped to model input aspect ratio
        cam_config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}, # Typical camera output
            raw={"size": (640, 480)}, # Can be adjusted if raw needed
            display="main" # Ensure 'main' stream is used for preview/capture_array
        )
        picam2.configure(cam_config)
        picam2.start()
        print("Camera started.")
        time.sleep(1) # Give camera a moment to start

        frame_count, start_time = 0, time.time()

        print("Running real-time depth estimation. Press Ctrl+C to quit.")
        while running:
            # Capture frame from Picamera2 (returns BGR numpy array)
            frame_original_bgr = picam2.capture_array()

            # Crop the frame to a square, centered, to match expected model input
            h_orig, w_orig, _ = frame_original_bgr.shape
            crop_size = min(h_orig, w_orig)
            start_x_crop = (w_orig - crop_size) // 2
            start_y_crop = (h_orig - crop_size) // 2
            frame_cropped_bgr = frame_original_bgr[start_y_crop:start_y_crop+crop_size, 
                                                   start_x_crop:start_x_crop+crop_size]

            # Preprocess the frame for the model
            # The preprocess_frame function expects RGB as input for its internal conversion,
            # so we convert from BGR to RGB before passing it.
            x_norm, raw_img_rgb_for_display = preprocess_frame(frame_cropped_bgr, 
                                                               MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x_norm, axis=0))
            interpreter.invoke()
            pred_depth = interpreter.get_tensor(output_details[0]['index'])
            pred_depth = np.squeeze(pred_depth, axis=0).squeeze(-1)

            # Ensure depth map is resized to MODEL_INPUT_HEIGHT x MODEL_INPUT_WIDTH
            if pred_depth.shape != (MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH):
                pred_depth = cv2.resize(pred_depth, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), 
                                        interpolation=cv2.INTER_CUBIC)

            aligned_depth = m * pred_depth + c

            # Generate visualizations (these functions produce RGB images)
            # pred_vis_rgb = plot_depth_with_points_fixed(pred_depth, points, "Predicted Depth", 
            #                                             cmap='plasma', H=MODEL_INPUT_HEIGHT, 
            #                                             W=MODEL_INPUT_WIDTH)
            aligned_vis_rgb = plot_depth_with_points_fixed(aligned_depth, points, "Aligned Depth", 
                                                            cmap='magma', H=MODEL_INPUT_HEIGHT, 
                                                            W=MODEL_INPUT_WIDTH)

            # Resize original image to match visualization dimensions
            original_vis_rgb = cv2.resize(raw_img_rgb_for_display, 
                                          (DISPLAY_WIDTH_SINGLE, DISPLAY_HEIGHT_SINGLE))

            # Combine visualizations horizontally
            combined_display_rgb = np.hstack([
                original_vis_rgb,
                #pred_vis_rgb,
                aligned_vis_rgb
            ])

            # Draw the combined image to the framebuffer
            display.draw(combined_display_rgb)

            frame_count += 1
            if (time.time() - start_time) > 1.5:
                fps = frame_count / (time.time() - start_time)
                print(f"FPS: {fps:.2f}")
                frame_count, start_time = 0, time.time()

    except Exception as e:
        import traceback
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
    finally:
        print("\nInitiating final resource cleanup...")
        if picam2 and picam2.started:
            picam2.stop()
            picam2.close()
            print("Camera stopped and closed.")
        if display:
            display.close()
        print("Cleanup complete.")

if __name__ == '__main__':
    # Ensure matplotlib is used in headless mode
    import matplotlib
    matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plot generation
    import matplotlib.pyplot as plt # Import plt after setting backend
    main()
