import numpy as np
import pandas as pd


def selectivity(acc_task: np.ndarray, acc_control: np.ndarray) -> np.ndarray:
    if acc_task.shape != acc_control.shape:
        raise ValueError("Accuracy arrays must have the same shape.")
    return acc_task - acc_control

def relative_depth(layer: int | np.ndarray, num_layers: int) -> float | np.ndarray:

    if num_layers <= 0:
        raise ValueError(f"Variable \"num_layers\" should be grater than 0.")
    
    if not isinstance(layer, (int, np.ndarray, np.integer)):
        raise TypeError(f"Variable \"layer\" should be an integer or a NumPy array.")
    
    if isinstance(layer, np.ndarray):
        if not np.issubdtype(layer.dtype, np.integer):
            raise TypeError("Variable \"layer\" must contain integer values.")
        
    if np.any(layer <= 0) or np.any(layer > num_layers):  # common check for both int and NumPy arrays
        raise ValueError("Each layer has to satisfy the condition: 1 <= layer <= num_layers.")

    return layer / num_layers

def center_of_gravity(sel: np.ndarray, *, normalize: bool = True, clip_negative: bool = True) -> float:
    # sel has L + 1 entries (layers 0..L); only layers 1..L are summed; returns nan if the clipped sum is 0
    pass

def peak_layer(sel: np.ndarray) -> int:
    pass

def summarize_probing(results: pd.DataFrame) -> pd.DataFrame:
    pass
