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
        
        self.optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': 1e-5},  # Very small lr for pre-trained
            {'params': new_params, 'lr': 1e-4}         # Larger lr for new layers
        ], weight_decay=0.01)  # Added weight decay
        
        self.button_criterion = nn.MSELoss()
        self.analog_criterion = nn.L1Loss()
        
        # Initialize tensors on device
        self.screen_tensor = torch.zeros((1, 3, 480, 854), 
                                       dtype=torch.float32, 
                                       device=self.device)
        self.control_tensor = torch.zeros((1, 12), 
                                        dtype=torch.float32, 
                                        device=self.device)
        
        # Loss weights
        self.button_weight = 1.0
        self.analog_weight = 2.0  # Increased from 0.5
        
        # Logging
        self.running_loss = []
        self.button_accuracy = []
        self.analog_error = []
        self.log_interval = 100
        
    def is_default_state(self, controller_state):
        """Check if the controller is in the default state"""
        # Check buttons (first 4 values)
        buttons_default = all(abs(x) < 0.01 for x in controller_state[:4])
        
        # Check analog sticks (values 4-7)
        # Consider 0.4-0.6 as the "dead zone" or default position
        sticks_default = all(0.4 <= x <= 0.6 for x in controller_state[4:8])
        
        # Check triggers (values 8-11)
        triggers_default = all(abs(x) < 0.01 for x in controller_state[8:])
        
        return buttons_default and sticks_default and triggers_default
    
    def train_step(self, screen_state, controller_state):
        # Check if controller is in default state
        if self.is_default_state(controller_state):
            return None  # Skip training
        
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
        
        # Store previous prediction if it exists
        if hasattr(self, 'prev_prediction'):
            prev_pred = self.prev_prediction
        else:
            prev_pred = None
            
        # Forward pass
        predictions = self.model(self.screen_tensor)
        self.prev_prediction = predictions.detach()  # Store for next step
        
        # Split predictions and targets
        button_pred = predictions[:, :4]
        button_target = self.control_tensor[:, :4]
        analog_pred = predictions[:, 4:]
        analog_target = self.control_tensor[:, 4:]
        
        # Compute weighted losses
        button_loss = self.button_criterion(button_pred, button_target) * self.button_weight
        analog_loss = self.analog_criterion(analog_pred, analog_target) * self.analog_weight
        
        # Add prediction difference loss if we have a previous prediction
        if prev_pred is not None:
            # Penalize if prediction doesn't change when input does
            target_diff = torch.abs(self.control_tensor - self.prev_control).mean()
            pred_diff = torch.abs(predictions - prev_pred).mean()
            diff_loss = torch.abs(target_diff - pred_diff) * 0.5
            total_loss = button_loss + analog_loss + diff_loss
        else:
            total_loss = button_loss + analog_loss
            
        self.prev_control = self.control_tensor.detach()
        
        # Compute metrics
        with torch.no_grad():
            button_acc = ((button_pred > 0.5) == (button_target > 0.5)).float().mean()
            analog_diff = torch.abs(analog_pred - analog_target)
            analog_err = analog_diff.mean()
            
            # Log max analog difference to detect large errors
            max_analog_err = analog_diff.max().item()
            
            metrics = {
                'btn_acc': button_acc.item(),
                'analog_err': analog_err.item(),
                'max_analog_err': max_analog_err,
                'loss': total_loss.item()
            }
            
            if len(self.running_loss) >= self.log_interval:
                print(f"\rLoss: {metrics['loss']:.4f} | "
                      f"Btn Acc: {metrics['btn_acc']:.1%} | "
                      f"Analog Err: {metrics['analog_err']:.3f} | "
                      f"Max Err: {metrics['max_analog_err']:.3f}", end="")
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return metrics