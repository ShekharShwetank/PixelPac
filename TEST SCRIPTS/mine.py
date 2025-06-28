#!/usr/bin/env python3

# File: realtime_depth.py (v7 - BGR Native Pipeline)
# Purpose: Real-time monocular depth estimation with a BGR-native pipeline
#          to match run.py logic and fix color issues permanently.

import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import time
import os
import sys
import signal
import mmap
import fcntl

from picamera2 import Picamera2

# ======================================================================================
# FrameBuffer Display Class for Console-based Real-Time Output
# ======================================================================================
class FrameBuffer:
    """A class to handle displaying NumPy arrays on the Linux framebuffer."""
    def __init__(self, device='/dev/fb0'):
        self.fb_dev = os.open(device, os.O_RDWR)
        var_info_bytes = fcntl.ioctl(self.fb_dev, 0x4600, b'\0' * 160)
        self.width = int.from_bytes(var_info_bytes[0:4], 'little')
        self.height = int.from_bytes(var_info_bytes[4:8], 'little')
        self.bpp = int.from_bytes(var_info_bytes[24:28], 'little')
        self.screen_size = self.width * self.height * (self.bpp // 8)
        self.fb_map = mmap.mmap(self.fb_dev, self.screen_size, mmap.MAP_SHARED, mmap.PROT_WRITE)
        print(f"Framebuffer initialized: {self.width}x{self.height} @ {self.bpp}bpp")

    def draw(self, frame_rgb):
        """Draw a single RGB frame to the framebuffer."""
        resized_frame = cv2.resize(frame_rgb, (self.width, self.height))
        
        if self.bpp == 32:
            output_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGRA)
        elif self.bpp == 16:
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

# ======================================================================================
# Main Application
# ======================================================================================

# --- Configuration ---
MODEL_PATH = "model.tflite"
MODEL_INPUT_SHAPE = (240, 240)
DISPLAY_HEIGHT = MODEL_INPUT_SHAPE[0]
DISPLAY_WIDTH = MODEL_INPUT_SHAPE[1]

# --- Global Objects & State for Graceful Shutdown ---
picam2, interpreter, display = None, None, None
running = True 

def cleanup_handler(signum=None, frame=None):
    """Signal handler to initiate a clean shutdown."""
    global running
    if running:
        print("\nShutdown signal received. Exiting loop...")
        running = False

def main():
    global picam2, interpreter, display, running
    
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    try:
        display = FrameBuffer()
        print(f"Loading model: {MODEL_PATH}")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        depth_output_index = -1
        for i, detail in enumerate(output_details):
            if "Identity_1" in detail['name']:
                depth_output_index = detail['index']
                break
        
        if depth_output_index == -1:
            raise RuntimeError("Could not find the 'Identity_1' output tensor in the model.")
        print(f"Found depth map output tensor at index {depth_output_index}.")

        picam2 = Picamera2()
        # --- THE FIX: Request BGR888 format to align with OpenCV's native format. ---
        cam_config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
        picam2.configure(cam_config)
        picam2.start()
        print("Camera started.")
        time.sleep(1) 

        frame_count, start_time = 0, time.time()
        
        while running:
            # capture_array() will now return a 3-channel BGR numpy array
            frame_original_bgr = picam2.capture_array()

            h, w, _ = frame_original_bgr.shape
            crop_size = min(h, w)
            start_x, start_y = (w - crop_size) // 2, (h - crop_size) // 2
            frame_cropped_bgr = frame_original_bgr[start_y:start_y+crop_size, start_x:start_x+crop_size]

            # Convert BGR to RGB *only* for the model input, as in run.py
            frame_rgb_for_model = cv2.cvtColor(frame_cropped_bgr, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(cv2.resize(frame_rgb_for_model, MODEL_INPUT_SHAPE), axis=0).astype(np.float32) / 255.0
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            raw_output = interpreter.get_tensor(depth_output_index)
            depth_map = np.squeeze(raw_output)

            depth_min, depth_max = depth_map.min(), depth_map.max()
            depth_vis_gray = (255 * (depth_map - depth_min) / (depth_max - depth_min + 1e-6)).astype(np.uint8)
            
            # applyColorMap creates a BGR image, which is what we want.
            depth_vis_bgr = cv2.applyColorMap(depth_vis_gray, cv2.COLORMAP_JET)

            # Use the original BGR frame for the left-side display.
            # original_vis_resized = cv2.resize(frame_cropped_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            # depth_vis_resized = cv2.resize(depth_vis_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            # Both images are now consistently BGR.
            # side_by_side_bgr = np.hstack((original_vis_resized, depth_vis_resized))
            
            # Draw the final BGR image.
            # display.draw(side_by_side_bgr)
            
            # Convert BGR to RGB for display
            original_vis_resized = cv2.resize(frame_cropped_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            original_vis_rgb = cv2.cvtColor(original_vis_resized, cv2.COLOR_BGR2RGB)
            depth_vis_resized = cv2.resize(depth_vis_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            depth_vis_rgb = cv2.cvtColor(depth_vis_resized, cv2.COLOR_BGR2RGB)
            
            # Create side-by-side with RGB images
            side_by_side_rgb = np.hstack((original_vis_rgb, depth_vis_rgb))
            
            # Draw the RGB image
            display.draw(side_by_side_rgb)


            frame_count += 1
            if (time.time() - start_time) > 1.5:
                fps = frame_count / (time.time() - start_time)
                print(f"FPS: {fps:.2f}")
                frame_count, start_time = 0, time.time()

    except Exception as e:
        import traceback
        print(f"\nAn error occurred in the main loop: {e}")
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
    main()