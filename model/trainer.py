import torch
import torch.nn as nn
import torch.optim as optim

class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Pre-allocate tensors for single sample
        self.screen_tensor = torch.zeros((1, 3, 240, 320), dtype=torch.float32)
        self.control_tensor = torch.zeros((1, 12), dtype=torch.float32)
        
    def train_step(self, screen_state, controller_state):
        # Ensure screen_state has batch dimension
        if screen_state.dim() == 3:
            screen_state = screen_state.unsqueeze(0)
        
        # Copy inputs to pre-allocated tensors
        self.screen_tensor.copy_(screen_state)
        self.control_tensor.copy_(torch.tensor(controller_state).unsqueeze(0))
        
        # Train immediately on this sample
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.model(self.screen_tensor)
        loss = self.criterion(predictions, self.control_tensor)
        
        # Print some debug info occasionally
        if torch.rand(1).item() < 0.01:  # 1% of the time
            print(f"\nDebug - Predictions: {predictions.detach().numpy().flatten()}")
            print(f"Debug - Controller: {controller_state}")
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()