import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

class ModelTrainer:
    def __init__(self, model, buffer_size=1000, batch_size=8):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.experience_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.update_frequency = 4
        self.frame_counter = 0
        
    def train_step(self, screen_state, controller_state):
        # Add to experience buffer
        self.experience_buffer.append((screen_state, controller_state))
        
        # Only train periodically
        self.frame_counter += 1
        if self.frame_counter % self.update_frequency != 0:
            return None
            
        # Only train if we have enough samples
        if len(self.experience_buffer) < self.batch_size:
            return None
            
        # Sample batch
        batch = random.sample(self.experience_buffer, self.batch_size)
        screens, controls = zip(*batch)
        
        # Convert to tensors (using full precision)
        screen_tensor = torch.cat([s for s in screens], dim=0)
        control_tensor = torch.FloatTensor([list(c.values()) for c in controls])
        
        # Train
        self.optimizer.zero_grad()
        predictions = self.model(screen_tensor)
        loss = self.criterion(predictions, control_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()