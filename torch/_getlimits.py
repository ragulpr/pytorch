from collections import namedtuple
import torch

__all__ = [
    'finfo'
]

# This follows semantics of numpy.finfo.
_Finfo = namedtuple('_Finfo', ['eps', 'tiny'])
_FINFO = {
    torch.HalfStorage: _Finfo(eps=0.00097656, tiny=6.1035e-05),
    torch.FloatStorage: _Finfo(eps=1.19209e-07, tiny=1.17549e-38),
    torch.DoubleStorage: _Finfo(eps=2.22044604925e-16, tiny=2.22507385851e-308),
    torch.cuda.HalfStorage: _Finfo(eps=0.00097656, tiny=6.1035e-05),
    torch.cuda.FloatStorage: _Finfo(eps=1.19209e-07, tiny=1.17549e-38),
    torch.cuda.DoubleStorage: _Finfo(eps=2.22044604925e-16, tiny=2.22507385851e-308),
}


def finfo(tensor):
    r"""
    Return floating point info about a `Tensor`:
    - `.eps` is the smallest number that can be added to 1 without being lost.
    - `.tiny` is the smallest positive number greater than zero
      (much smaller than `.eps`).

    Args:
        tensor (Tensor): tensor of floating point data.
    Returns:
        _Finfo: a `namedtuple` with fields `.eps` and `.tiny`.
    """
    return _FINFO[tensor.storage_type()]
