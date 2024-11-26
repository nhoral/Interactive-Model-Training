import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

class ModelTrainer:
    def __init__(self, model, buffer_size=256, batch_size=1):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.experience_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.update_frequency = 1
        self.frame_counter = 0
        
        self.screen_batch = torch.zeros((1, 3, 240, 320), dtype=torch.float32)
        self.control_batch = torch.zeros((1, 12), dtype=torch.float32)
        
    def train_step(self, screen_state, controller_state):
        if screen_state.dim() == 3:
            screen_state = screen_state.unsqueeze(0)
            
        self.experience_buffer.append((screen_state, controller_state))
        
        self.frame_counter += 1
        if self.frame_counter % self.update_frequency != 0 or len(self.experience_buffer) < self.batch_size:
            return None
            
        batch = random.sample(self.experience_buffer, self.batch_size)
        for i, (screen, control) in enumerate(batch):
            self.screen_batch[i].copy_(screen.squeeze(0))
            self.control_batch[i].copy_(torch.tensor(control))
        
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.model(self.screen_batch)
        loss = self.criterion(predictions, self.control_batch)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()