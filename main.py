import torch
from data_collection.controller_input import ControllerInput
from data_collection.screen_capture import ScreenCapture
from model.network import GameInputNetwork
from model.trainer import ComplexityAwareTrainer
import keyboard
import time
import argparse
import os
import tkinter as tk
from tkinter import ttk
import pyautogui  # or use pynput for more complex input simulation
import vgamepad as vg

class TrainingWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Training Monitor")
        self.root.attributes('-topmost', True)
        self.root.geometry('1430x145')  # Slightly wider to accommodate all inputs
        
        # Style
        style = ttk.Style()
        style.configure('TLabel', font=('Consolas', 10))
        
        # Create frame for metrics
        metrics_frame = ttk.Frame(self.root)
        metrics_frame.pack(fill='x', padx=5, pady=2)
        
        self.metrics_label = ttk.Label(metrics_frame, style='TLabel')
        self.metrics_label.pack(anchor='w')
        
        # Create frame for controller state
        controller_frame = ttk.Frame(self.root)
        controller_frame.pack(fill='x', padx=5, pady=2)
        
        self.controller_label = ttk.Label(controller_frame, style='TLabel')
        self.controller_label.pack(anchor='w')
        
        self.update_counter = 0
        self.update_frequency = 2  # Update display every N frames
        
    def update_display(self, timings, metrics, controller_state, mode='training', prediction=None):
        self.update_counter += 1
        if self.update_counter % self.update_frequency != 0:
            return
        """Update the display with timing, metrics, and controller state"""
        # Calculate total time from components
        total_time = sum(timings.values())
        
        # Format metrics line with mode indicator
        if mode == 'training':
            metrics_str = f"[TRAINING] Loss: {metrics['loss']:.4f} | Loop: {total_time:.1f}ms" if metrics else "Loss: N/A"
        else:
            metrics_str = f"[AI CONTROL] Loop: {total_time:.1f}ms"
        
        # Use prediction values in AI control mode, otherwise use controller_state
        state_to_display = prediction[0] if mode == 'prediction' and prediction is not None else controller_state
        
        # Format controller state line
        face_buttons = "".join([
            f"{'A' if state_to_display[0] > 0.5 else '_'}",  # A (BTN_SOUTH)
            f"{'B' if state_to_display[1] > 0.5 else '_'}",  # B (BTN_EAST)
            f"{'Y' if state_to_display[3] > 0.5 else '_'}",  # Y (BTN_NORTH) - swapped with X
            f"{'X' if state_to_display[2] > 0.5 else '_'}"   # X (BTN_WEST) - swapped with Y
        ])
        
        shoulder_buttons = "".join([
            f"{'LB' if state_to_display[10] > 0.5 else '__'}",
            f"{'RB' if state_to_display[11] > 0.5 else '__'}"
        ])
        
        stick_buttons = "".join([
            f"{'L3' if state_to_display[12] > 0.5 else '__'}",
            f"{'R3' if state_to_display[13] > 0.5 else '__'}"
        ])
        
        triggers = f"LT:{state_to_display[8]:.1f} RT:{state_to_display[9]:.1f}"
        left_stick = f"L({state_to_display[4]:.2f},{state_to_display[5]:.2f})"
        right_stick = f"R({state_to_display[6]:.2f},{state_to_display[7]:.2f})"
        
        controller_str = f"[{face_buttons}] [{shoulder_buttons}] [{stick_buttons}] {triggers} | {left_stick} | {right_stick}"
        
        # Update labels
        self.metrics_label.config(text=metrics_str)
        self.controller_label.config(text=controller_str)
        
        # Process pending events
        self.root.update()

def simulate_inputs(prediction, virtual_controller):
    """Simulate controller inputs based on model prediction"""
    pred = prediction[0]  # Get first batch item
    
    # Buttons (first 4 values)
    buttons = pred[:4] > 0.5
    if buttons[0]: virtual_controller.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    else: virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    
    if buttons[1]: virtual_controller.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    else: virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    
    if buttons[2]: virtual_controller.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
    else: virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
    
    if buttons[3]: virtual_controller.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
    else: virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
    
    # Analog sticks (values 4-7)
    # Convert from 0-1 range to -32768 to 32767 range
    left_x = int((pred[4] * 2 - 1) * 32767)
    left_y = int((pred[5] * 2 - 1) * 32767)
    right_x = int((pred[6] * 2 - 1) * 32767)
    right_y = int((pred[7] * 2 - 1) * 32767)
    
    virtual_controller.left_joystick(x_value=left_x, y_value=left_y)
    virtual_controller.right_joystick(x_value=right_x, y_value=right_y)
    
    # Triggers (values 8-9)
    # Convert from 0-1 range to 0-255 range
    left_trigger = int(pred[8] * 255)
    right_trigger = int(pred[9] * 255)
    
    virtual_controller.left_trigger(value=left_trigger)
    virtual_controller.right_trigger(value=right_trigger)
    
    # Shoulder buttons (values 10-11)
    if pred[10] > 0.5: virtual_controller.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
    else: virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
    
    if pred[11] > 0.5: virtual_controller.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    else: virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    
    # L3/R3 buttons (values 12-13)
    if pred[12] > 0.5: virtual_controller.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)
    else: virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)
    
    if pred[13] > 0.5: virtual_controller.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB)
    else: virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB)

def send_neutral_input(virtual_controller):
    """Send neutral input to the virtual controller"""
    # Release all buttons
    virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
    virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
    
    # Set sticks to neutral position
    virtual_controller.left_joystick(x_value=0, y_value=0)
    virtual_controller.right_joystick(x_value=0, y_value=0)
    
    # Set triggers to neutral position
    virtual_controller.left_trigger(value=0)
    virtual_controller.right_trigger(value=0)
    
    # Release L3/R3
    virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)
    virtual_controller.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB)
    
    # Update the virtual controller
    virtual_controller.update()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing: {device.type.upper()}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model to predict controller inputs from screen capture.')
    parser.add_argument('--reset', action='store_true', help='Start with a fresh model, ignoring any saved weights')
    parser.add_argument('--debug', action='store_true', help='Enable performance debugging')
    args = parser.parse_args()

    # Initialize components
    WIDTH = 256  # Reduced resolution for R3D processing
    HEIGHT = 256 # Square resolution works better with R3D
    NUM_FRAMES = 4  # Number of frames to maintain in buffer
    
    controller = ControllerInput()
    screen_capture = ScreenCapture(width=WIDTH, height=HEIGHT)
    model = GameInputNetwork(
        input_channels=3, 
        input_height=HEIGHT, 
        input_width=WIDTH,
        num_frames=NUM_FRAMES
    )
    model = model.to(device)
    
    # Load existing model unless reset is specified
    model_path = 'game_input_model.pth'
    if not args.reset and os.path.exists(model_path):
        print("Loading existing model from", model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Starting with fresh model")
    
    trainer = ComplexityAwareTrainer(model)
    
    # Create monitoring window
    window = TrainingWindow()
    print("Starting input prediction system...")
    print("Press 'Q' to stop training")
    
    mode = 'training'
    virtual_controller = None
    
    try:
        while True:
            loop_start = time.time()
            timings = {}
            
            # Keyboard check timing
            t0 = time.time()
            if keyboard.is_pressed('q'):
                print("\nStopping training (Q pressed)...")
                break
            
            if keyboard.is_pressed('t'):
                if mode == 'prediction':
                    # Send neutral input before switching to training mode
                    if virtual_controller is not None:
                        send_neutral_input(virtual_controller)
                
                mode = 'prediction' if mode == 'training' else 'training'
                print(f"\nSwitched to {mode} mode")
                
                if mode == 'prediction' and virtual_controller is None:
                    virtual_controller = vg.VX360Gamepad()
                
                time.sleep(0.5)
            timings['keyboard'] = (time.time() - t0) * 1000
            
            # Screen capture timing
            t0 = time.time()
            screen_tensor = screen_capture.get_frame()  # Already a tensor on GPU
            timings['screen'] = (time.time() - t0) * 1000
            
            # Controller timing
            t0 = time.time()
            controller_state = controller.get_state()
            timings['controller'] = (time.time() - t0) * 1000
            
            # Training/prediction timing
            t0 = time.time()
            if mode == 'training':
                metrics = trainer.train_step(screen_tensor, controller_state)
                timings['training'] = (time.time() - t0) * 1000
            else:
                with torch.no_grad():
                    prediction = model(screen_tensor)  # Removed unsqueeze(0) since screen_tensor is already batched
                    # Simulate the predicted inputs
                    simulate_inputs(prediction, virtual_controller)
                    virtual_controller.update()  # Important: update the virtual controller
                timings['prediction'] = (time.time() - t0) * 1000
            
            # Display update timing
            t0 = time.time()
            if mode == 'training':
                window.update_display(timings, metrics, controller_state, mode=mode)
            else:
                # Pass the prediction tensor instead of controller_state when in prediction mode
                window.update_display(timings, None, controller_state, mode='prediction', prediction=prediction.cpu().numpy())
            timings['display'] = (time.time() - t0) * 1000
            
            # Print timing breakdown
            print("\rTimings (ms) - ", end="")
            for key, value in timings.items():
                print(f"{key}: {value:.1f} | ", end="")
            
            # Adjust sleep time to maintain consistent frame rate
            time.sleep(max(0.016 - (time.time() - loop_start), 0))  # Target ~60 FPS
            
    finally:
        print("\nSaving model...")
        torch.save(model.state_dict(), model_path)
        print("Training stopped")
        window.root.destroy()

if __name__ == "__main__":
    main()