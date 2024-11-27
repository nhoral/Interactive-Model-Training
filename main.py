import torch
from data_collection.controller_input import ControllerInput
from data_collection.screen_capture import ScreenCapture
from model.network import GameInputNetwork
from model.trainer import ModelTrainer
import keyboard
import time
import argparse
import os
import tkinter as tk
from tkinter import ttk

class TrainingWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Training Monitor")
        self.root.attributes('-topmost', True)
        self.root.geometry('500x150')
        
        # Style
        style = ttk.Style()
        style.configure('TLabel', font=('Courier', 10))
        
        # Create labels
        self.timing_label = ttk.Label(self.root, style='TLabel')
        self.timing_label.pack(pady=5)
        
        self.loss_label = ttk.Label(self.root, style='TLabel')
        self.loss_label.pack(pady=5)
        
        self.input_label = ttk.Label(self.root, style='TLabel')
        self.input_label.pack(pady=5)
        
    def update_display(self, timings, metrics, controller_state):
        """Update the display with timing, metrics, and controller state"""
        # Format timing string
        timing_str = f"Total Time: {timings['total']:.1f}ms"
        
        # Format loss string
        loss_str = f"Loss: {metrics['loss']:.4f}" if metrics else "Loss: N/A"
        
        # Format controller input string
        buttons = "".join(["[X]" if state > 0.5 else "[ ]" for state in controller_state[:4]])
        analogs = ", ".join([f"{state:.2f}" for state in controller_state[4:8]])
        input_str = f"Buttons: {buttons} / Sticks: {analogs}"
        
        # Update labels
        self.timing_label.config(text=timing_str)
        self.loss_label.config(text=loss_str)
        self.input_label.config(text=input_str)
        
        # Process pending events
        self.root.update()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing: {device.type.upper()}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model to predict controller inputs from screen capture.')
    parser.add_argument('--reset', action='store_true', help='Start with a fresh model, ignoring any saved weights')
    args = parser.parse_args()

    # Initialize components
    WIDTH = 854  # 480p width
    HEIGHT = 480 # 480p height
    
    controller = ControllerInput()
    screen_capture = ScreenCapture(width=WIDTH, height=HEIGHT)
    model = GameInputNetwork(input_channels=3, input_height=HEIGHT, input_width=WIDTH)
    model = model.to(device)
    
    # Load existing model unless reset is specified
    model_path = 'game_input_model.pth'
    if not args.reset and os.path.exists(model_path):
        print("Loading existing model from", model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Starting with fresh model")
    
    trainer = ModelTrainer(model)
    
    # Create monitoring window
    window = TrainingWindow()
    print("Starting input prediction system...")
    print("Press 'Q' to stop training")
    
    try:
        while True:
            timings = {}
            loop_start = time.time()
            
            if keyboard.is_pressed('q'):
                print("\nStopping training (Q pressed)...")
                break
            
            # Time screen capture
            t0 = time.time()
            screen_state = screen_capture.get_frame()
            timings['screen'] = (time.time() - t0) * 1000
            
            # Get controller input
            controller_state = controller.get_state()
            
            # Convert to tensor and move to device
            screen_tensor = torch.from_numpy(screen_state).float().to(device)
            
            # Time model training
            t0 = time.time()
            metrics = trainer.train_step(screen_tensor, controller_state)
            timings['training'] = (time.time() - t0) * 1000
            
            # Time prediction
            t0 = time.time()
            with torch.no_grad():
                prediction = model(screen_tensor.unsqueeze(0))
            timings['prediction'] = (time.time() - t0) * 1000
            
            # Total loop time
            timings['total'] = (time.time() - loop_start) * 1000
            
            # Update display with metrics
            window.update_display(timings, metrics, controller_state)
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping training (Ctrl+C pressed)...")
    
    finally:
        print("\nSaving model...")
        torch.save(model.state_dict(), model_path)
        print("Training stopped")
        window.root.destroy()

if __name__ == "__main__":
    main()