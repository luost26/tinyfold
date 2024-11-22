from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch


_SupportedDType: TypeAlias = type[np.float32]


def export_tensor(x: np.ndarray | torch.Tensor, path: str | Path, dtype: _SupportedDType = np.float32) -> None:
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    with open(path, "wb") as f:
        x = x.astype(dtype)
        x.tofile(f)


def export_tensor_dict(
    x: Mapping[str, np.ndarray | torch.Tensor],
    dirpath: str | Path,
    dtype: _SupportedDType = np.float32,
) -> None:
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    for key, value in x.items():
        export_tensor(value, dirpath / f"{key}.bin", dtype=dtype)


def export_value_list(x: Sequence[int | float | str], path: str | Path) -> None:
    with open(path, "w") as f:
        for value in x:
            f.write(f"{value}\n")
