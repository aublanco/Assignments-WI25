import numpy as np


# Quantize the state space and action space
def quantize_state(state: dict, state_bins: dict) -> tuple:
    """
    
    Given the state and the bins for each state variable, quantize the state space.

    Args:
        state (dict): The state to be quantized.
        state_bins (dict): The bins used for quantizing each dimension of the state.

    Returns:
        tuple: The quantized representation of the state.
    """
    state_index = []
    for key in sorted(state.keys()):
        for v, b in zip(state[key], state_bins[key]):
            b_arr = np.asarray(b).flatten()
            bin_index = int(np.digitize(v, b_arr))
            state_index.append(bin_index)
    return tuple(state_index)

def quantize_action(action: float, bins: list) -> int:
    """
    Quantize the action based on the provided bins. 
    """
    return int(np.digitize(action, bins)) -1


