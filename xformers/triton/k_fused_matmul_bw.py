# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch
import triton
import triton.language as tl


# fmt: off
@triton.heuristics({
    'EVEN_N': lambda *args, **meta: args[3] % (meta['BLOCK_COL']) == 0,
})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_COL": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_COL": 64}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_COL": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_COL": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_COL": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_COL": 1024}, num_stages=3, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def kernel_bw(
    # Pointers to matrices
    GRAD_ACT, GRAD_OUT, ACT_INPUTS,
    # Matrix dimensions
    N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_gom, stride_aim,
    # Meta-parameters
    **META,
):
    # fmt: on

    """
    Go over all the activation inputs, compute the corresponding gradient
    """

    # extract metaparameters
    BLOCK_N = META["BLOCK_COL"]

    # this kernel is relatively simple in terms of scheduling:
    # - per row (pid_m)
    # - each program a given chunk on the col axis,
    # since it's more effective memory and occupancy wise
    pid_m, pid_n = tl.program_id(axis=0), tl.program_id(axis=1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # the memory addresses of elements in the first block of
    # A and W can be computed using numpy-style broadcasting
    act_input_ptrs = ACT_INPUTS + pid_m * stride_aim + rn

    # compute the gradient which is related to this activation
    if META["EVEN_N"]:
        act_in = tl.load(act_input_ptrs)
    else:
        act_in = tl.load(act_input_ptrs, mask=rn < N, other=0.0)

    grad_act = META["ACTIVATION_GRAD"](act_in)

    # now read the incoming gradient, the backpropagated one is the multiple of both
    if META["ACTIVATION_REQUIRES_INPUT"]:
        grad_out_ptrs = GRAD_OUT + pid_m * stride_gom + rn
        if META["EVEN_N"]:
            grad_out = tl.load(grad_out_ptrs)
        else:
            grad_out = tl.load(grad_out_ptrs, mask=rn < N)

        grad_act *= grad_out

    # write back result
    grad_act_ptrs = GRAD_ACT + pid_m * stride_gom + rn
    tl.store(grad_act_ptrs, grad_act, mask=rn < N)


def fused_matmul_backward(
    grad_out: torch.Tensor,
    inputs: torch.Tensor,
    act_in: Optional[torch.Tensor],
    weight: torch.Tensor,
    trainable_weight: bool,
    trainable_bias: bool,
    activation_grad=None,
    act_requires_input: bool = True
):
    """
    Compute grad_in = activation^-1(grad_out) @ weight.transpose()

    .. note: The weight buffer is transposed on the fly
    .. note: Activation gradient needs to be a Triton kernel
    """

    # Make sure that we don't have to handle the stride over cols
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    grad_out_ = grad_out if grad_out.ndim == 2 else grad_out.flatten(0, 1)
    inputs_ = inputs if inputs.ndim == 2 else inputs.flatten(0, 1)

    assert grad_out_.shape[1] == weight.shape[0], "Incompatible dimensions in between grad_out and weight"

    M, N = grad_out_.shape
    N, _ = weight.shape

    # Compute the gradient for the activation
    if activation_grad is not None:
        grad_act = torch.empty_like(grad_out_)

        # Some activations do not require their inputs to
        # know of their grad, the downstream grad is enough
        if act_in is None:
            act_in = grad_out_

        def grid(META):
            return (
                M,
                triton.cdiv(N, META["BLOCK_COL"]),
            )

        # fmt: off
        kernel_bw[grid](
            # data ptrs
            grad_act, grad_out_, act_in,
            # shapes
            N,
            # strides
            grad_act.stride(0), act_in.stride(0),
            weight.stride(0), weight.stride(1),
            # optional fused activation
            ACTIVATION_GRAD=activation_grad,
            ACTIVATION_REQUIRES_INPUT=act_requires_input
        )

        # Backpropagation going up, the reference gradient is now
        # just before the activation
        grad_out_ = grad_act

    # The following ops can also be handled by triton
    grad_in = grad_out_ @ weight
    grad_weight = grad_out_.transpose(1, 0) @ inputs_ if trainable_weight else None
    grad_bias = torch.sum(grad_out_, 0) if trainable_bias else None

    return grad_in.reshape_as(inputs), grad_weight, grad_bias
