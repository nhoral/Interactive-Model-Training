import torchvision.models as models
import torch.nn as nn
import torch

class GameInputNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=480, input_width=854):
        super(GameInputNetwork, self).__init__()
        
        # Load pre-trained MobileNetV3-Small
        self.features = models.mobilenet_v3_small(weights='DEFAULT').features
        
        # Freeze only the first few layers
        for param in list(self.features.parameters())[:4]:  # Keep most layers trainable
            param.requires_grad = False
            
        # Lightweight heads for our specific task
        self.shared_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(576, 128),  # MobileNetV3-Small outputs 576 features
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Separate paths for buttons and analog
        self.button_path = nn.Sequential(
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        
        self.analog_path = nn.Sequential(
            nn.Linear(128, 8),
            # Custom activation to ensure analog starts at 0.5
            nn.Sigmoid()
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    if m in self.analog_path:
                        # Initialize analog outputs to center position
                        nn.init.constant_(m.bias, 0.5)
                    else:
                        nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Ensure input is float and normalized
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
            
        # Extract features
        features = self.features(x)
        shared = self.shared_head(features)
        
        # Get predictions
        button_out = self.button_path(shared)
        analog_out = self.analog_path(shared)
        
        return torch.cat((button_out, analog_out), dim=1)

    def get_trainable_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)