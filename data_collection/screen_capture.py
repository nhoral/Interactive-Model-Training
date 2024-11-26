import numpy as np
import win32gui
import win32ui
import win32con
import win32api
import cv2

class ScreenCapture:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        
        # Get handle to the primary monitor
        self.hwnd = win32gui.GetDesktopWindow()
        
        # Get screen dimensions
        self.screen_width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        self.screen_height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        
        # Calculate capture region (center of screen)
        self.capture_x = self.screen_width // 4
        self.capture_y = self.screen_height // 4
        self.capture_width = self.screen_width // 2
        self.capture_height = self.screen_height // 2
        
        # Create device context and bitmap
        self.hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        self.mfc_dc = win32ui.CreateDCFromHandle(self.hwnd_dc)
        self.save_dc = self.mfc_dc.CreateCompatibleDC()
        
        self.bitmap = win32ui.CreateBitmap()
        self.bitmap.CreateCompatibleBitmap(self.mfc_dc, self.capture_width, self.capture_height)
        self.save_dc.SelectObject(self.bitmap)
        
        # Pre-allocate numpy arrays
        self.frame_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.output_buffer = np.zeros((3, self.height, self.width), dtype=np.float32)
        
    def get_frame(self):
        # Capture screen region
        self.save_dc.BitBlt(
            (0, 0), 
            (self.capture_width, self.capture_height), 
            self.mfc_dc, 
            (self.capture_x, self.capture_y), 
            win32con.SRCCOPY
        )
        
        # Convert bitmap to numpy array
        bmp_bits = self.bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmp_bits, dtype=np.uint8)
        img = img.reshape((self.capture_height, self.capture_width, 4))[:, :, :3]  # Remove alpha channel
        
        # Resize and normalize in one step
        np.divide(
            cv2.resize(img, (self.width, self.height), dst=self.frame_buffer),
            255.0,
            out=self.frame_buffer
        )
        
        # Transpose for PyTorch (CHW format)
        # Copy directly to the correct shape
        np.copyto(self.output_buffer, self.frame_buffer.transpose(2, 0, 1))
        
        return self.output_buffer
        
    def __del__(self):
        # Clean up resources
        self.save_dc.DeleteDC()
        self.mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.hwnd_dc)
        win32gui.DeleteObject(self.bitmap.GetHandle())