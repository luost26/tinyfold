import torch
import numpy as np


def export_tensor(x: np.ndarray | torch.Tensor, path: str, dtype: type[np.dtype] = np.float32) -> None:
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    with open(path, 'wb') as f:
        x = x.astype(dtype)
        x.tofile(f)
    return x
