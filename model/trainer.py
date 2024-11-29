import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ModelTrainer:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing: {self.device.type.upper()}")
        
        self.model = model.to(self.device)
        
        # Initialize previous states
        self.prev_control = None
        self.prev_prediction = None
        
        # Different learning rates for pre-trained and new layers
        pretrained_params = list(model.features.parameters())
        new_params = list(model.button_path.parameters()) + list(model.analog_path.parameters())
        
        self.optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': 1e-5},
            {'params': new_params, 'lr': 1e-4}
        ], weight_decay=0.01)
        
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
        
        # Use mixed precision training with new API
        self.scaler = torch.amp.GradScaler()
        
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
            return None
        
        # Handle screen tensor
        if not isinstance(screen_state, torch.Tensor):
            self.screen_tensor = torch.from_numpy(screen_state).float().to(self.device)
        else:
            self.screen_tensor = screen_state.float().to(self.device)
        
        # Handle controller state
        if isinstance(controller_state, list):
            self.control_tensor = torch.tensor([controller_state], 
                                             dtype=torch.float32, 
                                             device=self.device)
        elif isinstance(controller_state, np.ndarray):
            self.control_tensor = torch.from_numpy(controller_state).float().to(self.device)
        elif isinstance(controller_state, torch.Tensor):
            self.control_tensor = controller_state.float().to(self.device)
        
        # Ensure tensors have correct shape
        if self.screen_tensor.dim() == 3:
            self.screen_tensor = self.screen_tensor.unsqueeze(0)
        if self.control_tensor.dim() == 1:
            self.control_tensor = self.control_tensor.unsqueeze(0)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        predictions = self.model(self.screen_tensor)
        
        # Split predictions and targets
        button_pred = predictions[:, :4]
        button_target = self.control_tensor[:, :4]
        analog_pred = predictions[:, 4:]
        analog_target = self.control_tensor[:, 4:]
        
        # Compute weighted losses
        button_loss = self.button_criterion(button_pred, button_target) * self.button_weight
        analog_loss = self.analog_criterion(analog_pred, analog_target) * self.analog_weight
        
        # Initialize prev_control if it's None
        if self.prev_control is None:
            self.prev_control = self.control_tensor.detach()
            self.prev_prediction = predictions.detach()
            return None  # Skip first training step
            
        # Compute prediction difference loss
        target_diff = torch.abs(self.control_tensor - self.prev_control).mean()
        pred_diff = torch.abs(predictions - self.prev_prediction).mean()
        diff_loss = torch.abs(target_diff - pred_diff) * 0.5
        
        # Update previous states
        self.prev_control = self.control_tensor.detach()
        self.prev_prediction = predictions.detach()
        
        total_loss = button_loss + analog_loss + diff_loss
        
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
        
        # Backward pass without gradient scaling
        total_loss.backward()
        self.optimizer.step()
        
        # Update model's frame buffer
        self.model.update_frame_buffer(self.screen_tensor[0])
        
        # Only proceed with training if we have enough frames
        if self.model.frame_buffer[0].sum() == 0:  # Check if buffer is still initializing
            return None
        
        return metrics

class ComplexityAwareTrainer(ModelTrainer):
    def __init__(self, model):
        super().__init__(model)
        
        # Buffer for recent states to detect patterns
        self.state_buffer = []
        self.buffer_size = 60  # 1 second at 60fps
        
        # Exponential moving average for complexity baseline
        self.complexity_ema = 0.0
        self.ema_alpha = 0.01  # Small alpha for stable baseline
        
    def calculate_complexity(self, state):
        """Calculate complexity score for current state"""
        complexity = 0.0
        
        # Button complexity (simultaneous presses)
        buttons_active = sum(1 for x in state[:4] if x > 0.5)
        complexity += buttons_active * 0.3
        
        # Stick movement (deviation from center)
        stick_movement = sum(abs(x - 0.5) for x in state[4:8])
        complexity += stick_movement * 0.4
        
        # Trigger usage
        trigger_usage = sum(state[8:10])
        complexity += trigger_usage * 0.2
        
        # L3/R3 usage (if implemented)
        if len(state) > 12:  # Check if L3/R3 are present
            thumb_buttons = sum(1 for x in state[12:14] if x > 0.5)
            complexity += thumb_buttons * 0.1
        
        return complexity
    
    def calculate_sample_weight(self, state):
        """Calculate importance weight for training sample"""
        # Get current complexity
        current_complexity = self.calculate_complexity(state)
        
        # Update exponential moving average
        self.complexity_ema = (self.ema_alpha * current_complexity + 
                             (1 - self.ema_alpha) * self.complexity_ema)
        
        # Calculate weight based on:
        # 1. How much more complex than baseline
        # 2. Sudden changes in complexity
        complexity_ratio = (current_complexity + 0.1) / (self.complexity_ema + 0.1)
        
        # Update state buffer for pattern detection
        self.state_buffer.append(state)
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
        
        # Detect sudden changes if we have enough history
        change_multiplier = 1.0
        if len(self.state_buffer) >= 10:
            recent_complexity = np.mean([self.calculate_complexity(s) 
                                       for s in self.state_buffer[-10:]])
            if current_complexity > recent_complexity * 1.5:
                change_multiplier = 1.5  # Boost weight for sudden complexity increases
        
        # Combine factors and clip weight to reasonable range
        weight = complexity_ratio * change_multiplier
        return np.clip(weight, 0.5, 3.0)
    
    def train_step(self, screen_state, controller_state):
        """Enhanced training step with complexity-based weighting"""
        # Skip if in default state
        if self.is_default_state(controller_state):
            return None
            
        # Calculate sample weight
        sample_weight = self.calculate_sample_weight(controller_state)
        
        # Process tensors (using parent class methods)
        if not isinstance(screen_state, torch.Tensor):
            self.screen_tensor = torch.from_numpy(screen_state).float().to(self.device)
        else:
            self.screen_tensor = screen_state.float().to(self.device)
        
        if isinstance(controller_state, list):
            self.control_tensor = torch.tensor([controller_state], 
                                             dtype=torch.float32, 
                                             device=self.device)
        elif isinstance(controller_state, np.ndarray):
            self.control_tensor = torch.from_numpy(controller_state).float().to(self.device)
        elif isinstance(controller_state, torch.Tensor):
            self.control_tensor = controller_state.float().to(self.device)
        
        # Ensure tensors have correct shape
        if self.screen_tensor.dim() == 3:
            self.screen_tensor = self.screen_tensor.unsqueeze(0)
        if self.control_tensor.dim() == 1:
            self.control_tensor = self.control_tensor.unsqueeze(0)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        predictions = self.model(self.screen_tensor)
        
        # Split predictions and targets
        button_pred = predictions[:, :4]
        button_target = self.control_tensor[:, :4]
        analog_pred = predictions[:, 4:]
        analog_target = self.control_tensor[:, 4:]
        
        # Apply sample weight to losses
        button_loss = self.button_criterion(button_pred, button_target) * self.button_weight * sample_weight
        analog_loss = self.analog_criterion(analog_pred, analog_target) * self.analog_weight * sample_weight
        
        # Initialize prev_control if it's None
        if self.prev_control is None:
            self.prev_control = self.control_tensor.detach()
            self.prev_prediction = predictions.detach()
            return None  # Skip first training step
            
        # Compute prediction difference loss
        target_diff = torch.abs(self.control_tensor - self.prev_control).mean()
        pred_diff = torch.abs(predictions - self.prev_prediction).mean()
        diff_loss = torch.abs(target_diff - pred_diff) * 0.5 * sample_weight
        
        # Update previous states
        self.prev_control = self.control_tensor.detach()
        self.prev_prediction = predictions.detach()
        
        total_loss = button_loss + analog_loss + diff_loss
        
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
                'loss': total_loss.item(),
                'weight': sample_weight
            }
            
            if len(self.running_loss) >= self.log_interval:
                print(f"\rLoss: {metrics['loss']:.4f} | "
                      f"Weight: {metrics['weight']:.2f} | "
                      f"Btn Acc: {metrics['btn_acc']:.1%} | "
                      f"Analog Err: {metrics['analog_err']:.3f} | "
                      f"Max Err: {metrics['max_analog_err']:.3f}", end="")
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Update model's frame buffer
        self.model.update_frame_buffer(self.screen_tensor[0])
        
        # Only proceed with training if we have enough frames
        if self.model.frame_buffer[0].sum() == 0:  # Check if buffer is still initializing
            return None
        
        return metrics