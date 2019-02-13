# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from mtorch.simple_parser import read_softmax_tree

import smt_cuda
import smt_cpu


class SoftmaxTreeFunction(Function):
    @staticmethod
    def forward(ctx, x, group_offsets, group_sizes, axis):
        assert 0 <= axis < x.dim(), "invalid axis for x of size: {}".format(x.size())
        node_count = group_offsets[-1] + group_sizes[-1]
        assert x.size(axis) == node_count, "Channel count: {} must match tree node count: {}".format(
            x.size(axis), node_count
        )
        if x.is_cuda:
            smt_ = smt_cuda
        else:
            smt_ = smt_cpu
        prob = smt_.forward(x, group_offsets, group_sizes, axis)[0]

        ctx.softmax_axis = axis
        ctx.save_for_backward(prob, group_offsets, group_sizes)
        return prob

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_group_offsets = grad_group_sizes = grad_axis = None
        if ctx.needs_input_grad[0]:
            axis = ctx.softmax_axis
            prob, group_offsets, group_sizes = ctx.saved_tensors
            if prob.is_cuda:
                smt_ = smt_cuda
            else:
                smt_ = smt_cpu
            grad_x = smt_.backward(
                prob, grad_output,
                group_offsets, group_sizes,
                axis
            )[0]

        return grad_x, grad_group_offsets, grad_group_sizes, grad_axis


class SoftmaxTree(nn.Module):
    def __init__(self, tree, axis=1):
        """SoftmaxTree is multiple softmaxes with an inherent tree relation assumed between softmax groups
        :param tree: path to the tree file (format as in Yolo)
        :param axis: axis to apply softmax (jagged axis)
        """
        super(SoftmaxTree, self).__init__()
        self.tree = tree  # type: str
        self.axis = axis

        group_offsets, group_sizes, cid_groups, parents, _, _, _ = read_softmax_tree(self.tree)
        self.register_buffer('group_offsets', torch.from_numpy(np.array(group_offsets, dtype=np.int32)))
        self.register_buffer('group_sizes', torch.from_numpy(np.array(group_sizes, dtype=np.int32)))
        self.node_count = len(cid_groups)
        self.group_count = len(group_offsets)
        assert self.node_count == group_offsets[-1] + group_sizes[-1], "node count: {} last group: {}+{}".format(
            self.node_count, group_offsets[-1], group_sizes[-1]
        )

    def forward(self, x):
        return SoftmaxTreeFunction.apply(
            x,
            self.group_offsets, self.group_sizes, self.axis
        )

    def extra_repr(self):
        """Extra information
        """
        return 'tree={}, nodes={}, groups={}{}'.format(
            self.tree, self.node_count, self.group_count, ", axis={}".format(self.axis) if self.axis != 1 else ""
        )


# TODO: make these a proper unit test
if __name__ == '__main__':
    from StringIO import StringIO

    # Create a flat softmax
    net = SoftmaxTree(StringIO("boz -1\nbozak -1\ngoat -1\n"))

    # create a matrix 4x3x8 so that softmax will be applied to the second dimension
    a = torch.rand(4, 3, 8)
    b = net(a)
    # total should be 1.0
    print(b[0, :, 0].sum())

    # now test with cuda
    net = net.cuda()
    a = a.cuda()
    b = net(a)
    print(b[0, :, 0].sum())
