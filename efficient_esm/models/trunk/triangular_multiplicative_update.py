# mypy: ignore-errors
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partialmethod

import torch
import torch.nn as nn


def permute_final_dims(tensor: torch.Tensor, inds: Sequence[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class BaseTriangleMultiplicativeUpdate(nn.Module, ABC):
    """
    Implements Algorithms 11 and 12.
    """

    @abstractmethod
    def __init__(self, c_z, c_hidden, _outgoing):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_g = nn.Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = nn.Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = nn.LayerNorm(self.c_z)
        self.layer_norm_out = nn.LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        p = torch.matmul(a, b)

        return permute_final_dims(p, (1, 2, 0))

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """


class TriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements Algorithms 11 and 12.
    """

    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super().__init__(c_z=c_z, c_hidden=c_hidden, _outgoing=_outgoing)

        self.linear_a_p = nn.Linear(self.c_z, self.c_hidden)
        self.linear_a_g = nn.Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = nn.Linear(self.c_z, self.c_hidden)
        self.linear_b_g = nn.Linear(self.c_z, self.c_hidden, init="gating")

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask
        a = a * self.sigmoid(self.linear_a_g(z))
        a = a * self.linear_a_p(z)
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)

        x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)
