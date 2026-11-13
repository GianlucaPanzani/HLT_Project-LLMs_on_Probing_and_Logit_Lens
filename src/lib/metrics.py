import numpy as np
import pandas as pd


def selectivity(acc_task: np.ndarray, acc_control: np.ndarray) -> np.ndarray:
    return acc_task - acc_control

def relative_depth(layer: int | np.ndarray, num_layers: int) -> float | np.ndarray:
    pass

def center_of_gravity(sel: np.ndarray, *, normalize: bool = True, clip_negative: bool = True) -> float:
    # sel has L + 1 entries (layers 0..L); only layers 1..L are summed; returns nan if the clipped sum is 0
    pass

def peak_layer(sel: np.ndarray) -> int:
    pass

def summarize_probing(results: pd.DataFrame) -> pd.DataFrame:
    pass
