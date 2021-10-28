# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import revlib
import torch.nn as nn
from revlib import ReversibleSequential as RevSeq


def ReversibleSequence(blocks: nn.ModuleList):
    return RevSeq(*[f for layer in blocks for f in layer],
                  split_dim=2, memory_mode=revlib.MemoryModes.autograd_function)  # or autograd_graph
