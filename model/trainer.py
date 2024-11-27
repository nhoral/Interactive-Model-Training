import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ModelTrainer:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing: {self.device.type.upper()}")
        
        self.model = model.to(self.device)
        
        # Different learning rates for pre-trained and new layers
        pretrained_params = list(model.features.parameters())
        new_params = list(model.button_path.parameters()) + list(model.analog_path.parameters())
        
        self.optimizer = optim.Adam([
            {'params': pretrained_params, 'lr': 1e-5},  # Very small lr for pre-trained
            {'params': new_params, 'lr': 1e-4}         # Larger lr for new layers
        ])
        
        self.button_criterion = nn.BCELoss()
        self.analog_criterion = nn.MSELoss()
        
        # Initialize tensors on device
        self.screen_tensor = torch.zeros((1, 3, 480, 854), 
                                       dtype=torch.float32, 
                                       device=self.device)
        self.control_tensor = torch.zeros((1, 12), 
                                        dtype=torch.float32, 
                                        device=self.device)
        
    def train_step(self, screen_state, controller_state):
        # Handle screen tensor
        if not isinstance(screen_state, torch.Tensor):
            self.screen_tensor = torch.from_numpy(screen_state).to(self.device)
        else:
            self.screen_tensor = screen_state.to(self.device)
        
        # Handle controller state
        if isinstance(controller_state, list):
            self.control_tensor = torch.tensor([controller_state], 
                                             dtype=torch.float32, 
                                             device=self.device)
        elif isinstance(controller_state, np.ndarray):
            self.control_tensor = torch.from_numpy(controller_state).to(self.device)
        elif isinstance(controller_state, torch.Tensor):
            self.control_tensor = controller_state.to(self.device)
        
        # Ensure tensors have correct shape
        if self.screen_tensor.dim() == 3:
            self.screen_tensor = self.screen_tensor.unsqueeze(0)
        if self.control_tensor.dim() == 1:
            self.control_tensor = self.control_tensor.unsqueeze(0)
        
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.model(self.screen_tensor)
        
        # Split predictions and targets
        button_pred = predictions[:, :4]
        button_target = self.control_tensor[:, :4]
        analog_pred = predictions[:, 4:]
        analog_target = self.control_tensor[:, 4:]
        
        # Compute losses
        button_loss = self.button_criterion(button_pred, button_target)
        analog_loss = self.analog_criterion(analog_pred, analog_target)
        
        # Combined loss
        loss = button_loss + analog_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()