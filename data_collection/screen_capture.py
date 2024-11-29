"""
Screen Capture using DXCam-CPP for efficient DirectX-based capture
"""

import numpy as np
import torch
import cv2
import time
import dxcam_cpp as dxcam

class ScreenCapture:
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height
        
        print("\nInitializing DXCamera...")
        
        try:
            # Create camera with video mode enabled and capture region matching target size
            self.camera = dxcam.create(
                max_buffer_len=2,  # Minimize buffer size
                region=(0, 0, width, height)  # Capture at target size directly
            )
            print("Camera created")
            
            # Start capture in video mode for consistent frame rate
            print("Starting capture in video mode...")
            self.camera.start(target_fps=60, video_mode=True)
            
            # Test capture and setup
            test_frame = self.camera.get_latest_frame()
            if test_frame is None:
                raise RuntimeError("Failed to get test frame")
            
            # Setup GPU
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Pre-allocate GPU tensors
            self.frame_tensor = torch.zeros(
                (1, 4, height, width),  # BGRA format
                dtype=torch.uint8,
                device=self.device
            )
            self.screen_tensor = torch.zeros(
                (1, 3, height, width),
                dtype=torch.float32,
                device=self.device
            )
            
            # Create dedicated CUDA stream
            self.stream = torch.cuda.Stream() if self.device.type == 'cuda' else None
            
            print(f"Initialization complete")
            
        except Exception as e:
            print(f"ERROR during initialization: {str(e)}")
            if hasattr(self, 'camera'):
                self.camera.stop()
            raise
    
    def get_frame(self):
        with torch.cuda.stream(self.stream) if self.stream else torch.no_grad():
            # Get latest frame (should never be None in video mode)
            frame = self.camera.get_latest_frame()
            
            if frame is not None:
                # Copy frame directly to GPU
                self.frame_tensor.copy_(
                    torch.from_numpy(frame).to(
                        self.device, 
                        non_blocking=True
                    ).permute(2, 0, 1).unsqueeze(0)
                )
                
                # Convert BGRA to RGB and normalize on GPU
                self.screen_tensor.copy_(
                    self.frame_tensor[:, [2, 1, 0]].float() / 255.0  # BGR -> RGB and normalize
                )
            
            return self.screen_tensor
    
    def __del__(self):
        if hasattr(self, 'camera'):
            print("\nStopping DXCamera...")
            self.camera.stop()