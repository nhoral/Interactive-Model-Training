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
        self.root.attributes('-topmost', True)  # Stay on top
        self.root.geometry('400x100')  # Set window size
        
        # Style
        style = ttk.Style()
        style.configure('TLabel', font=('Courier', 10))
        
        # Create labels
        self.timing_label = ttk.Label(self.root, style='TLabel')
        self.timing_label.pack(pady=10)
        
        self.loss_label = ttk.Label(self.root, style='TLabel')
        self.loss_label.pack(pady=10)
        
        # Initialize animation state
        self.tick = False
    
    def update_display(self, timings, loss):
        """Update the display with new timing and loss information"""
        self.tick = not self.tick
        indicator = "|" if self.tick else "-"
        
        # Format timing string
        important_timings = {k: timings[k] for k in ['screen', 'training', 'prediction', 'total'] 
                           if k in timings}
        timing_str = " | ".join([f"{k}: {v:.1f}ms" for k, v in important_timings.items()])
        
        # Format loss string
        loss_str = f"Loss: {loss:.4f}" if loss is not None else ""
        
        # Update labels
        self.timing_label.config(text=f"{indicator} {timing_str}")
        self.loss_label.config(text=loss_str)
        
        # Process any pending events
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
            
            # Get controller input (not timed)
            controller_state = controller.get_state()
            
            # Convert to tensor and move to device
            screen_tensor = torch.from_numpy(screen_state).float().to(device)
            
            # Time model training
            t0 = time.time()
            loss = trainer.train_step(screen_tensor, controller_state)
            timings['training'] = (time.time() - t0) * 1000
            
            # Time prediction
            t0 = time.time()
            with torch.no_grad():
                prediction = model(screen_tensor.unsqueeze(0))
            timings['prediction'] = (time.time() - t0) * 1000
            
            # Total loop time
            timings['total'] = (time.time() - loop_start) * 1000
            
            # Update display
            window.update_display(timings, loss)
            
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