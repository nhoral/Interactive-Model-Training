from inputs import get_gamepad

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
        self._state_list = [0] * 12  # Pre-allocated list for faster access
    
    def get_state(self):
        """Returns the current state of all controller inputs"""
        try:
            events = get_gamepad()
            for event in events:
                if event.code in self.button_states:
                    self.button_states[event.code] = event.state
        except Exception as e:
            print(f"Controller read error: {e}")
        
        # Update pre-allocated list
        for i, value in enumerate(self.button_states.values()):
            self._state_list[i] = value
            
        return self._state_list  # Return list instead of dict for faster processing

    def __del__(self):
        """Cleanup when the object is destroyed"""
        pass