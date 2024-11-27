import numpy as np
import win32gui
import win32ui
import win32con
import win32api
import cv2

class ScreenCapture:
    def __init__(self, width=854, height=480):
        self.width = width
        self.height = height
        
        # Get handle to the primary monitor
        self.hwnd = win32gui.GetDesktopWindow()
        
        # Get full screen dimensions
        self.screen_width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        self.screen_height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        
        # Create device context and bitmap for full screen
        self.hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        self.mfc_dc = win32ui.CreateDCFromHandle(self.hwnd_dc)
        self.save_dc = self.mfc_dc.CreateCompatibleDC()
        
        self.bitmap = win32ui.CreateBitmap()
        self.bitmap.CreateCompatibleBitmap(self.mfc_dc, self.screen_width, self.screen_height)
        self.save_dc.SelectObject(self.bitmap)
        
        # Pre-allocate numpy arrays
        self.frame_buffer = np.zeros((height, width, 3), dtype=np.float32)
        self.output_buffer = np.zeros((3, height, width), dtype=np.float32)
        
    def get_frame(self):
        # Capture entire screen
        self.save_dc.BitBlt(
            (0, 0), 
            (self.screen_width, self.screen_height), 
            self.mfc_dc, 
            (0, 0),
            win32con.SRCCOPY
        )
        
        # Convert bitmap to numpy array
        bmp_bits = self.bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmp_bits, dtype=np.uint8)
        img = img.reshape((self.screen_height, self.screen_width, 4))[:, :, :3]  # Remove alpha channel
        
        # Resize and normalize in one step
        np.divide(
            cv2.resize(img, (self.width, self.height), dst=self.frame_buffer),
            255.0,
            out=self.frame_buffer
        )
        
        # Basic augmentation
        img = self.frame_buffer
        
        # Random brightness adjustment (0.8 to 1.2)
        if np.random.random() < 0.3:
            brightness = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 1)
            
        # Random contrast adjustment
        if np.random.random() < 0.3:
            contrast = np.random.uniform(0.8, 1.2)
            img = np.clip((img - 0.5) * contrast + 0.5, 0, 1)
            
        # Normalize
        img = (img - img.mean()) / (img.std() + 1e-5)
        
        # Update output buffer
        np.copyto(self.output_buffer, img.transpose(2, 0, 1))
        return self.output_buffer
        
    def __del__(self):
        # Clean up resources
        self.save_dc.DeleteDC()
        self.mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.hwnd_dc)
        win32gui.DeleteObject(self.bitmap.GetHandle())