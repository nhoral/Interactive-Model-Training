import torch.nn as nn
import torch
from torchvision.models.video import r3d_18

class GameInputNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=256, input_width=256, num_frames=4):
        super(GameInputNetwork, self).__init__()
        
        # Get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained R3D-18
        self.features = r3d_18(weights='DEFAULT')
        
        # Remove the last classification layer
        self.features = nn.Sequential(*list(self.features.children())[:-1])
        
        # Freeze early layers
        for param in list(self.features.parameters())[:20]:  # Adjust freezing as needed
            param.requires_grad = False
            
        # Lightweight heads for our specific task
        self.shared_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),  # R3D-18 outputs 512 features
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Separate paths for buttons and analog
        self.button_path = nn.Sequential(
            nn.Linear(128, 6),  # Updated to 6 for A,B,X,Y,L3,R3
            nn.Sigmoid()
        )
        
        self.analog_path = nn.Sequential(
            nn.Linear(128, 8),  # Remains 8 for analog sticks and triggers
            nn.Sigmoid()
        )
        
        # Initialize frame buffer on the correct device
        self.frame_buffer = torch.zeros((num_frames, input_channels, input_height, input_width),
                                      device=self.device)
        self.num_frames = num_frames
        
        self._initialize_weights()
        
        # Use half precision for faster computation
        self.half_precision = False
        
        # Optimize for inference
        if not self.training:
            self.features = torch.jit.script(self.features)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    if m in self.analog_path:
                        nn.init.constant_(m.bias, 0.5)
                    else:
                        nn.init.constant_(m.bias, 0)
    
    def update_frame_buffer(self, new_frame):
        # Ensure new frame is on the correct device
        if new_frame.device != self.device:
            new_frame = new_frame.to(self.device)
            
        # Roll the buffer and add new frame
        self.frame_buffer = torch.roll(self.frame_buffer, -1, dims=0)
        self.frame_buffer[-1] = new_frame
        
    def forward(self, x):
        # Ensure input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Update frame buffer
        self.update_frame_buffer(x[0])
        
        # Prepare input for R3D (B, C, T, H, W)
        x = self.frame_buffer.unsqueeze(0)  # Add batch dimension
        x = x.permute(0, 2, 1, 3, 4)  # Reorder to (B, C, T, H, W)
        
        # Ensure input is float and normalized
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
            
        # Extract features
        features = self.features(x)
        shared = self.shared_head(features)
        
        # Add small random noise during training
        if self.training:
            shared = shared + torch.randn_like(shared) * 0.01
        
        # Get predictions
        button_out = self.button_path(shared)
        analog_out = self.analog_path(shared)
        
        # Ensure analog outputs aren't stuck in the middle
        if self.training:
            analog_bias = (analog_out - 0.5).sign() * 0.01
            analog_out = analog_out + analog_bias
            
        return torch.cat((button_out, analog_out), dim=1)

    def get_trainable_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)