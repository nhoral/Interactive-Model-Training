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
            'ABS_X': 0,      # Left stick X
            'ABS_Y': 0,      # Left stick Y
            'ABS_RX': 0,     # Right stick X
            'ABS_RY': 0,     # Right stick Y
            'ABS_Z': 0,      # Left trigger
            'ABS_RZ': 0,     # Right trigger
            'BTN_TL': 0,     # Left bumper
            'BTN_TR': 0      # Right bumper
        }
        self._state_list = [0] * 12  # Pre-allocated list
        
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
        """Returns the current state of all controller inputs"""
        # Update pre-allocated list with current state
        for i, value in enumerate(self.button_states.values()):
            self._state_list[i] = value
        return self._state_list
    
    def __del__(self):
        """Cleanup when the object is destroyed"""
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=1.0)