import torch
from data_collection.controller_input import ControllerInput
from data_collection.screen_capture import ScreenCapture
from model.network import GameInputNetwork
from model.trainer import ModelTrainer
import keyboard
import time

def print_timing(timing_dict):
    """Print timing information on one line"""
    timing_str = " | ".join([f"{k}: {v:.1f}ms" for k, v in timing_dict.items()])
    print(f"{timing_str}", end='\r')  # Use carriage return to update in place

def main():
    # Initialize components with smaller dimensions
    WIDTH = 320
    HEIGHT = 240
    
    controller = ControllerInput()
    screen_capture = ScreenCapture(width=WIDTH, height=HEIGHT)
    
    # Initialize model with same dimensions
    model = GameInputNetwork(input_channels=3, input_height=HEIGHT, input_width=WIDTH)
    trainer = ModelTrainer(model)
    
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
            
            # Time controller input
            t0 = time.time()
            controller_state = controller.get_state()
            timings['controller'] = (time.time() - t0) * 1000
            
            # Time tensor conversion
            t0 = time.time()
            screen_tensor = torch.FloatTensor(screen_state)
            if screen_tensor.dim() == 3:
                screen_tensor = screen_tensor.unsqueeze(0)
            timings['tensor_conv'] = (time.time() - t0) * 1000
            
            # Time model training
            t0 = time.time()
            loss = trainer.train_step(screen_tensor, controller_state)
            timings['training'] = (time.time() - t0) * 1000
            
            # Time prediction
            t0 = time.time()
            with torch.no_grad():
                prediction = model(screen_tensor)
            timings['prediction'] = (time.time() - t0) * 1000
            
            # Total loop time
            timings['total'] = (time.time() - loop_start) * 1000
            
            # Print timings
            print_timing(timings)
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping training (Ctrl+C pressed)...")
    
    finally:
        print("\nSaving model...")
        torch.save(model.state_dict(), 'game_input_model.pth')
        print("Training stopped")

if __name__ == "__main__":
    main()