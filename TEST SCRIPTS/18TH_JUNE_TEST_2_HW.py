#18TH_JUNE_TEST_2_HW.py
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import time
import os
import sys
import signal
import mmap
import fcntl
from threading import Thread, Event

from picamera2 import Picamera2

from gpiozero import LED, Buzzer

import matplotlib
matplotlib.use('Agg')
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
        """Unmap memory and close the framebuffer device.
           Also clears the screen to black before closing.
        """
        if hasattr(self, 'fb_map'):
            # Clear the framebuffer to black
            black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8) # Create a black RGB frame
            self.draw(black_frame) # Use the existing draw method to write black to screen
            self.fb_map.close()
        if hasattr(self, 'fb_dev'):
            os.close(self.fb_dev)
        print("Framebuffer closed and screen cleared.") # Updated message

# --- Global Objects & State for Graceful Shutdown ---
picam2, interpreter, display = None, None, None
green_led, red_led, buzzer = None, None, None
running = True
alert_active = False
alert_stop_event = Event()
alert_thread = None

def cleanup_handler(signum=None, frame=None):
    """Signal handler to initiate a clean shutdown."""
    global running
    if running:
        print("\nShutdown signal received. Exiting loop...")
        running = False

# --- Configuration ---
MODEL_PATH = "ADALITE_TFLITE.tflite"
MODEL_INPUT_HEIGHT = 256
MODEL_INPUT_WIDTH = 256
DISPLAY_WIDTH_SINGLE = 256
DISPLAY_HEIGHT_SINGLE = 256

# --- Hardware GPIO Pin Definitions (BCM numbering) ---
GREEN_LED_PIN = 27
RED_LED_PIN = 17
BUZZER_PIN = 22

# --- Alert System Configuration ---
DEPTH_ALERT_THRESHOLD = 3.0

DEPTH_ROI_X_RATIO = 0.3
DEPTH_ROI_Y_RATIO = 0.3
DEPTH_ROI_WIDTH_RATIO = 0.4
DEPTH_ROI_HEIGHT_RATIO = 0.4

LED_BLINK_ON_TIME = 0.1
LED_BLINK_OFF_TIME = 0.1
BUZZER_BEEP_ON_TIME = 0.15
BUZZER_BEEP_OFF_TIME = 0.15


# ======================================================================================
# Original Depth Estimation Functions (adapted for console environment)
# ======================================================================================

def preprocess_frame(frame, inputWidth, inputHeight):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (inputWidth, inputHeight), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    normalized_img = ((img_resized / 255.0 - mean) / std).astype(np.float32)
    return normalized_img, img_rgb

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

# ======================================================================================
# Hardware Alert Management Functions
# ======================================================================================
def alert_thread_function(current_red_led, current_buzzer, stop_event):
    """
    This function runs in a separate thread to handle flashing/beeping
    without blocking the main loop.
    """
    print("Alert thread started.")
    while not stop_event.is_set():
        current_red_led.on()
        current_buzzer.on()
        if stop_event.wait(timeout=LED_BLINK_ON_TIME):
            break

        current_red_led.off()
        current_buzzer.off()
        if stop_event.wait(timeout=LED_BLINK_OFF_TIME):
            break
            
    current_red_led.off()
    current_buzzer.off()
    print("Alert thread stopped.")

def update_alert_system_state():
    """Updates LEDs and Buzzer based on the global alert_active flag."""
    global alert_thread, alert_active, green_led, red_led, buzzer

    if alert_active:
        green_led.off()
        if alert_thread is None or not alert_thread.is_alive():
            alert_stop_event.clear()
            alert_thread = Thread(target=alert_thread_function, args=(red_led, buzzer, alert_stop_event))
            alert_thread.daemon = True
            alert_thread.start()
    else:
        if alert_thread is not None and alert_thread.is_alive():
            alert_stop_event.set()
            alert_thread.join(timeout=(LED_BLINK_ON_TIME + LED_BLINK_OFF_TIME) * 2)
            alert_thread = None
            
        red_led.off()
        buzzer.off()
        green_led.on()

def main():
    global picam2, interpreter, display, green_led, red_led, buzzer, running, alert_active

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    m, c = (-0.37965089082717896, 14.945058822631836)
    num_points_per_axis = 6

    xs = np.linspace(0, MODEL_INPUT_WIDTH - 1, num_points_per_axis, dtype=int)
    ys = np.linspace(0, MODEL_INPUT_HEIGHT - 1, num_points_per_axis, dtype=int)
    points = [(x, y) for y in ys for x in xs]

    try:
        display = FrameBuffer()

        print("Initializing LEDs and Buzzer...")
        green_led = LED(GREEN_LED_PIN)
        red_led = LED(RED_LED_PIN)
        buzzer = Buzzer(BUZZER_PIN)
        print("LEDs and Buzzer initialized.")
        
        green_led.on()
        red_led.off()
        buzzer.off()

        print("Loading TFLite model...")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Model loaded.")

        picam2 = Picamera2()
        cam_config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            raw={"size": (640, 480)},
            display="main"
        )
        picam2.configure(cam_config)
        picam2.start()
        print("Camera started.")
        time.sleep(1)

        frame_count, start_time = 0, time.time()
        min_depth_in_roi = float('inf')

        print("Running real-time depth estimation. Press Ctrl+C to quit.")
        while running:
            frame_original_bgr = picam2.capture_array()

            h_orig, w_orig, _ = frame_original_bgr.shape
            crop_size = min(h_orig, w_orig)
            start_x_crop = (w_orig - crop_size) // 2
            start_y_crop = (h_orig - crop_size) // 2
            frame_cropped_bgr = frame_original_bgr[start_y_crop:start_y_crop+crop_size, 
                                                   start_x_crop:start_x_crop+crop_size]

            x_norm, raw_img_rgb_for_display = preprocess_frame(frame_cropped_bgr, 
                                                               MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
            
            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x_norm, axis=0))
            interpreter.invoke()
            pred_depth = interpreter.get_tensor(output_details[0]['index'])
            pred_depth = np.squeeze(pred_depth, axis=0).squeeze(-1)

            if pred_depth.shape != (MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH):
                pred_depth = cv2.resize(pred_depth, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), 
                                        interpolation=cv2.INTER_CUBIC)

            aligned_depth = m * pred_depth + c

            # ================================================================
            # FIXED: Depth-based Alert Logic - Calculate min_depth_in_roi from aligned_depth
            # ================================================================
            depth_h, depth_w = aligned_depth.shape
            roi_x_start = int(depth_w * DEPTH_ROI_X_RATIO)
            roi_y_start = int(depth_h * DEPTH_ROI_Y_RATIO)
            roi_width = int(depth_w * DEPTH_ROI_WIDTH_RATIO)
            roi_height = int(depth_h * DEPTH_ROI_HEIGHT_RATIO)
            roi_x_end = roi_x_start + roi_width
            roi_y_end = roi_y_start + roi_height

            depth_roi = aligned_depth[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            if depth_roi.size > 0:
                min_depth_in_roi = np.min(depth_roi)
                if min_depth_in_roi < DEPTH_ALERT_THRESHOLD:
                    if not alert_active:
                        print(f"ALERT: Object detected at {min_depth_in_roi:.2f} < {DEPTH_ALERT_THRESHOLD:.2f}")
                    alert_active = True
                else:
                    if alert_active:
                        print(f"SAFE: Minimum depth {min_depth_in_roi:.2f} >= {DEPTH_ALERT_THRESHOLD:.2f}")
                    alert_active = False
            else:
                if alert_active:
                    print("ROI empty, reverting to safe state")
                alert_active = False
                min_depth_in_roi = float('inf')

            update_alert_system_state()
            # ================================================================

            aligned_vis_rgb = plot_depth_with_points_fixed(aligned_depth, points, 
                                                            f"Aligned Depth (Min ROI: {min_depth_in_roi:.2f})",
                                                            cmap='magma', H=MODEL_INPUT_HEIGHT, 
                                                            W=MODEL_INPUT_WIDTH)

            original_vis_rgb = cv2.resize(raw_img_rgb_for_display, 
                                          (DISPLAY_WIDTH_SINGLE, DISPLAY_HEIGHT_SINGLE))

            roi_draw_color = (0, 255, 255) if alert_active else (255, 255, 0)
            cv2.rectangle(aligned_vis_rgb, (roi_x_start, roi_y_start), 
                          (roi_x_end, roi_y_end), roi_draw_color, 2)
            
            status_text = "ALERT: PROXIMITY!" if alert_active else "SAFE"
            text_color = (255, 0, 0) if alert_active else (0, 255, 0)
            cv2.putText(original_vis_rgb, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            cv2.putText(original_vis_rgb, f"Min Depth: {min_depth_in_roi:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

            combined_display_rgb = np.hstack([
                original_vis_rgb,
                aligned_vis_rgb
            ])

            display.draw(combined_display_rgb)

            frame_count += 1
            if (time.time() - start_time) > 1.5:
                fps = frame_count / (time.time() - start_time)
                print(f"FPS: {fps:.2f}, Min Depth in ROI: {min_depth_in_roi:.2f}, Alert: {alert_active}")
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
            display.close() # This will now clear the screen
        if alert_thread is not None and alert_thread.is_alive():
            alert_stop_event.set()
            alert_thread.join(timeout=1.0)
        
        if green_led: green_led.close()
        if red_led: red_led.close()
        if buzzer: buzzer.close()
        print("Hardware (LEDs, Buzzer) closed.")
        print("Cleanup complete.")

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main()
(cat_venv) pi@depth-pi:~/CAT_TEST$ 

