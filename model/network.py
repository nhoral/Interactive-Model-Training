import torch
import torch.nn as nn

class GameInputNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=300, input_width=400):
        super(GameInputNetwork, self).__init__()
        
        # Store input dimensions
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        
        # Even more simplified CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=16, stride=8),  # Larger stride for faster processing
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the size of flattened features
        self._feature_size = self._get_conv_output_size()
        
        # Simplified output layers
        self.output_layer = nn.Sequential(
            nn.Linear(self._feature_size, 32),  # Reduced hidden size
            nn.ReLU(),
            nn.Linear(32, 12)  # 12 controller outputs
        )
        
        print(f"Feature size: {self._feature_size}")
        
    def _get_conv_output_size(self):
        x = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
        x = self.conv_layers(x)
        return x.shape[1]
        
    def forward(self, x):
        x = self.conv_layers(x)
        return self.output_layer(x)