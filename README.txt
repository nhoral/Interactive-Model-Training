# Game Input Prediction Model

This project allows a user to train a model while playing a game. The purpose of the model is to emulate the user and predict the next input based on screen pixels.

## Features
- Real-time capture of Xbox controller inputs
- Screen pixel capture at 60Hz (every 16ms)
- Neural network training to predict user inputs
- Live prediction capabilities

## Technical Requirements
- Python 3.8+
- PyTorch
- OpenCV (for screen capture)
- XInput (for Xbox controller input)
- NumPy

## Project Structure
project/
├── data_collection/
│ ├── controller_input.py
│ └── screen_capture.py
├── model/
│ ├── network.py
│ └── trainer.py
├── utils/
│ └── preprocessing.py
└── main.py

## Implementation Details
1. Data Collection
   - Xbox controller inputs captured using XInput
   - Screen pixels captured and preprocessed using OpenCV
   - Data synchronized and stored in memory buffer

2. Model Architecture
   - CNN-based architecture for processing screen pixels
   - LSTM layers for temporal dependencies
   - Output layer predicting controller states

3. Training Process
   - Real-time training during gameplay
   - Batch processing of recent experiences
   - Continuous model updates

## Getting Started
1. Install dependencies:
pip install torch opencv-python numpy xinput mss keyboard

2. Run the application:
python main.py