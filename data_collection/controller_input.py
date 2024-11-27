from inputs import get_gamepad
import threading
import time

class ControllerInput:
    def __init__(self):
        self.button_states = {
            'BTN_SOUTH': 0,  # A button
            'BTN_EAST': 0,   # B button
            'BTN_NORTH': 0,  # Y button
            'BTN_WEST': 0,   # X button
            'ABS_X': 0,      # Left stick X (centered)
            'ABS_Y': 0,      # Left stick Y (centered)
            'ABS_RX': 0,     # Right stick X (centered)
            'ABS_RY': 0,     # Right stick Y (centered)
            'ABS_Z': 0,      # Left trigger
            'ABS_RZ': 0,     # Right trigger
            'BTN_TL': 0,     # Left bumper
            'BTN_TR': 0      # Right bumper
        }
        self._state_list = [0] * 12
        
        # Start background thread to continuously update controller state
        self._running = True
        self._thread = threading.Thread(target=self._update_state)
        self._thread.daemon = True
        self._thread.start()
    
    def _update_state(self):
        """Background thread to continuously read controller state"""
        while self._running:
            try:
                events = get_gamepad()
                for event in events:
                    if event.code in self.button_states:
                        self.button_states[event.code] = event.state
            except Exception as e:
                pass  # Ignore errors in background thread
            time.sleep(0.001)  # Small sleep to prevent CPU thrashing
    
    def get_state(self):
        """Returns normalized controller inputs"""
        normalized_state = []
        
        for key, value in self.button_states.items():
            if key.startswith('BTN_'):
                # Buttons are already 0 or 1
                normalized_state.append(float(value))
            elif key in ['ABS_X', 'ABS_Y', 'ABS_RX', 'ABS_RY']:
                # Normalize analog sticks from -32768~32767 to 0~1
                normalized_state.append((value + 32768) / 65535)
            elif key in ['ABS_Z', 'ABS_RZ']:
                # Normalize triggers from 0~255 to 0~1
                normalized_state.append(value / 255)
        
        return normalized_state
    
    def __del__(self):
        """Cleanup when the object is destroyed"""
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=1.0)