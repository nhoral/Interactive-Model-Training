import cv2
import numpy as np

def preprocess_screen(frame, target_size=(800, 600)):
    """Preprocess screen capture for the neural network"""
    # Resize
    frame = cv2.resize(frame, target_size)
    
    # Normalize
    frame = frame / 255.0
    
    # Convert to torch format (C, H, W)
    frame = frame.transpose(2, 0, 1)
    
    return frame

def preprocess_controller(controller_state):
    """Convert controller state to normalized numpy array"""
    # Convert button states to array
    # Implement based on your controller input format
    pass