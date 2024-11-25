import cv2
import numpy as np
import mss

class ScreenCapture:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.sct = mss.mss()
        
    def get_frame(self):
        """Capture and preprocess a single frame"""
        # Capture the screen
        monitor = self.sct.monitors[1]  # Primary monitor
        screenshot = self.sct.grab(monitor)
        
        # Convert to numpy array
        frame = np.array(screenshot)
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        
        # Resize
        frame = cv2.resize(frame, (self.width, self.height))
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        # Ensure correct shape (C, H, W)
        frame = frame.transpose(2, 0, 1)
        
        return frame