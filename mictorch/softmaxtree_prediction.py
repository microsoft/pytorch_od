# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from mictorch.simple_parser import read_softmax_tree

import smtpred_cuda
import smtpred_cpu


def _find_max_stack_size(group_offsets, group_sizes, child, child_sizes, root_size, g=-1):
    if g == -1:
        max_stack_size = 0
        for g in range(root_size + 1):
            stack_size = _find_max_stack_size(group_offsets, group_sizes, child, child_sizes,
                                              root_size, g=g)
            if stack_size > max_stack_size:
                max_stack_size = stack_size
        return max_stack_size

    max_stack_size = 1
    offset = group_offsets[g]
    size = group_sizes[g]
    for n in range(offset, offset + size):
        g = child[n]
        if g < 0:
            continue
        stack_size = child_sizes[n] + _find_max_stack_size(group_offsets, group_sizes, child, child_sizes,
                                                           root_size, g=g)
        if stack_size > max_stack_size:
            max_stack_size = stack_size
    return max_stack_size


class SoftmaxTreePredictionFunction(Function):
    @staticmethod
    def forward(ctx,
                conf, obj,
                group_offsets, group_sizes, child, child_sizes,
                threshold, output_tree_path, append_max,
                root_size, stack_size,
                ):
        node_count = group_offsets[-1] + group_sizes[-1]
        assert conf.size(1) == node_count, "Channel count: {} must match tree node count: {}".format(
            conf.size(1), node_count
        )
        if conf.is_cuda:
            smtpred_ = smtpred_cuda
        else:
            smtpred_ = smtpred_cpu
        if obj is None:
            obj = torch.zeros(0).type_as(conf)
        else:
            # if objectness is provided
            assert conf.numel() / conf.size(1) == obj.numel(), "Invalid obj dimension"
        top_pred = smtpred_.forward(
            conf, obj,
            group_offsets, group_sizes, child, child_sizes,
            threshold, output_tree_path, append_max,
            root_size, stack_size
        )[0]

        return top_pred

    @staticmethod
    def backward(ctx, grad_output):
        return tuple([None] * 11)


class SoftmaxTreePrediction(nn.Module):
    def __init__(self, tree, threshold=0.5, append_max=True, output_tree_path=False):
        super(SoftmaxTreePrediction, self).__init__()
        self.tree = tree  # type: str
        self.threshold = threshold
        self.append_max = append_max
        self.output_tree_path = output_tree_path

        group_offsets, group_sizes, cid_groups, parents, child, child_sizes, self.root_size = read_softmax_tree(
            self.tree
        )
        self.stack_size = _find_max_stack_size(group_offsets, group_sizes, child, child_sizes, self.root_size)
        # TODO: share buffers with SoftmaxTree
        self.register_buffer('group_offsets', torch.from_numpy(np.array(group_offsets, dtype=np.int32)))
        self.register_buffer('group_sizes', torch.from_numpy(np.array(group_sizes, dtype=np.int32)))
        self.register_buffer('child', torch.from_numpy(np.array(child, dtype=np.int32)))
        self.register_buffer('child_sizes', torch.from_numpy(np.array(child_sizes, dtype=np.int32)))
        self.node_count = len(cid_groups)
        self.group_count = len(group_offsets)
        assert self.node_count == group_offsets[-1] + group_sizes[-1], "node count: {} last group: {}+{}".format(
            self.node_count, group_offsets[-1], group_sizes[-1]
        )

    def forward(self, conf, obj=None):
        if self.output_tree_path:
            assert obj is not None, "output_tree_path requires objectness bottom"
        return SoftmaxTreePredictionFunction.apply(
            conf, obj,
            self.group_offsets, self.group_sizes, self.child, self.child_sizes,
            self.threshold, self.output_tree_path, self.append_max,
            self.root_size, self.stack_size,
        )

    def extra_repr(self):
        """Extra information
        """
        return 'tree={}, nodes={}, groups={}{}{}{}{}'.format(
            self.tree, self.node_count, self.group_count,
            ", root_size={}".format(self.root_size) if self.root_size else "",
            ", stack_size={}".format(self.stack_size) if self.stack_size != 1 else "",
            ", append_max={}".format(self.append_max) if not self.append_max else "",
            ", output_tree_path={}".format(self.output_tree_path) if self.output_tree_path else "",
        )
