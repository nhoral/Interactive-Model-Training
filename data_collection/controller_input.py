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
            'BTN_TR': 0,     # Right bumper
            'BTN_THUMBL': 0, # Left stick click (L3)
            'BTN_THUMBR': 0  # Right stick click (R3)
        }
        self._state_list = [0] * 14
        
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
        
        # Order matters! This defines the indices in the state list
        button_order = [
            'BTN_SOUTH',    # A - index 0
            'BTN_EAST',     # B - index 1
            'BTN_WEST',     # X - index 2
            'BTN_NORTH',    # Y - index 3
            'ABS_X',        # Left stick X - index 4
            'ABS_Y',        # Left stick Y - index 5
            'ABS_RX',       # Right stick X - index 6
            'ABS_RY',       # Right stick Y - index 7
            'ABS_Z',        # Left trigger - index 8
            'ABS_RZ',       # Right trigger - index 9
            'BTN_TL',       # Left bumper - index 10
            'BTN_TR',       # Right bumper - index 11
            'BTN_THUMBL',   # L3 - index 12
            'BTN_THUMBR'    # R3 - index 13
        ]
        
        for key in button_order:
            value = self.button_states[key]
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