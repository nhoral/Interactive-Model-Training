# Game Input Prediction Model

Real-time model training system that learns to emulate player inputs from game screen pixels. Uses complexity-aware training to emphasize important gameplay moments.

## Core Components

### Input Processing
- Screen: DirectX capture at 60Hz (dxcam-cpp)
- Controller: Xbox input via 'inputs' library
- State format: [14 values]
  - 0-3: Face buttons (A,B,X,Y)
  - 4-7: Analog sticks (LX,LY,RX,RY)
  - 8-9: Triggers (LT,RT)
  - 10-11: Bumpers (LB,RB)
  - 12-13: Stick buttons (L3,R3)

### Neural Network
- Base: R3D-18 pretrained model
- Input: 4-frame buffer (3,4,256,256)
- Output: 14 normalized values [0-1]
- Training features:
  - Mixed precision
  - Frozen early layers
  - Separate button/analog paths
  - Dropout & noise injection

### Complexity-Aware Training
- Weights samples based on:
  - Button combinations
  - Stick movement magnitude
  - Trigger usage
  - Pattern recognition
  - Temporal changes
- EMA baseline for complexity
- Weight range: 0.5-3.0

### Performance
- Target loop: 16ms (60 FPS)
- CUDA acceleration
- DirectX screen capture
- Threaded controller input

### Key Dependencies
- PyTorch + CUDA
- dxcam-cpp
- inputs (controller)
- vgamepad (prediction mode)

## Usage Modes
1. Training: Learn from player inputs
2. Prediction: Emulate learned behavior via virtual controller

## File Structure
project/
├── data_collection/
│ ├── controller_input.py
│ └── screen_capture.py
├── model/
│ ├── network.py
│ └── trainer.py
└── main.py