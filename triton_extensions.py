# Extensions for Triton Language

import triton
import triton.language as core

# Ops for slicing (take/put) local tensor (extension to https://github.com/triton-lang/triton/pull/2715)
@triton.jit
def _indicator_(n_dims: core.constexpr, idx: core.constexpr, pos: core.constexpr, pos_dim: core.constexpr):
    core.static_assert(idx < n_dims)
    core.static_assert(pos < pos_dim)
    y = core.arange(0, pos_dim)
    y = core.where(y==pos, 1, 0)
    
    for n in core.static_range(0, n_dims):
        if n != n_dims - 1 - idx:
            y = core.expand_dims(y, n)
    return y

@triton.jit
def _take_slice_(x, n_dims: core.constexpr, idx: core.constexpr, pos: core.constexpr, pos_dim:core.constexpr, keep_dim: core.constexpr = True):
    ind = _indicator_(n_dims, idx, pos, pos_dim)
    y = core.sum(x * ind, n_dims - 1 - idx)
    if keep_dim:
        y = core.expand_dims(y, n_dims - 1 - idx)

    return y

# TODO T: note this op can use 1x tl.where in total instead of 2..
@triton.jit
def _put_slice_(x, n_dims: core.constexpr, idx: core.constexpr, pos: core.constexpr, pos_dim:core.constexpr, input_slice):
    ind = _indicator_(n_dims, idx, pos, pos_dim)
    y = core.where(ind==1, input_slice, x)
    return y