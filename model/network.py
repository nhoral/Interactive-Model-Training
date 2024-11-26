import torch
import torch.nn as nn

class GameInputNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=240, input_width=320):
        super(GameInputNetwork, self).__init__()
        
        # Store input dimensions
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        
        # More efficient CNN architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 4, kernel_size=16, stride=16, padding=0),  # Aggressive downsampling
            nn.ReLU(inplace=True),  # inplace operations
            nn.Conv2d(4, 8, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        self._feature_size = self._get_conv_output_size()
        
        # Single linear layer for faster inference
        self.output_layer = nn.Linear(self._feature_size, 12)
        
    def forward(self, x):
        return self.output_layer(self.conv_layers(x))
        
    def _get_conv_output_size(self):
        x = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
        x = self.conv_layers(x)
        return x.shape[1]