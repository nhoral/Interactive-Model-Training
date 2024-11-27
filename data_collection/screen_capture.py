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
        
        # Create device context and bitmap for target resolution
        self.hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        self.mfc_dc = win32ui.CreateDCFromHandle(self.hwnd_dc)
        self.save_dc = self.mfc_dc.CreateCompatibleDC()
        
        # Create bitmap at target resolution instead of full screen
        self.bitmap = win32ui.CreateBitmap()
        self.bitmap.CreateCompatibleBitmap(self.mfc_dc, width, height)
        self.save_dc.SelectObject(self.bitmap)
        
        # Pre-allocate numpy array (channels first for PyTorch)
        self.output_buffer = np.zeros((3, height, width), dtype=np.float32)
        
        # Get screen dimensions for scaling
        self.screen_width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        self.screen_height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        
    def get_frame(self):
        # Capture and resize in one step using StretchBlt
        self.save_dc.StretchBlt(
            (0, 0),
            (self.width, self.height),
            self.mfc_dc,
            (0, 0),
            (self.screen_width, self.screen_height),
            win32con.SRCCOPY
        )
        
        # Convert bitmap to numpy array
        bmp_bits = self.bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmp_bits, dtype=np.uint8)
        img = img.reshape((self.height, self.width, 4))[:, :, :3]  # Remove alpha channel
        
        # Normalize and transpose in one step
        np.divide(img, 255.0, out=self.output_buffer.reshape(self.height, self.width, 3))
        self.output_buffer = self.output_buffer.reshape(3, self.height, self.width)
        
        return self.output_buffer
        
    def __del__(self):
        self.save_dc.DeleteDC()
        self.mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.hwnd_dc)
        win32gui.DeleteObject(self.bitmap.GetHandle())