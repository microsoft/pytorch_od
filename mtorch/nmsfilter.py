# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.autograd import Function

import nmsfilt_cuda
import nmsfilt_cpu


class NMSFilterFunction(Function):
    @staticmethod
    def forward(ctx, bbs, conf,
                nms_threshold, classes, pre_threshold, first_class):
        if bbs.is_cuda:
            nms_ = nmsfilt_cuda
        else:
            nms_ = nmsfilt_cpu
        top_conf = nms_.forward(bbs, conf,
                                nms_threshold, classes, pre_threshold, first_class)[0]

        return top_conf

    @staticmethod
    def backward(ctx, grad_output):
        return tuple([None] * 6)


class NMSFilter(nn.Module):
    """Applies Non-Maximal Suppression filter to bounding box confidence values
    Each class (and each batch) will be filtered independently
    """
    def __init__(self, nms_threshold=0.45, classes=1, pre_threshold=-1.0, first_class=0):
        """NMSFilter
        :param nms_threshold: NMS threshold
        :param classes: number of classes to filter (-1 to filter all classes independently)
        :param pre_threshold: amplitude threshold to apply before NMS
        :param first_class: The first class to start filtering "classes"
        """
        super(NMSFilter, self).__init__()
        self.nms_threshold = nms_threshold
        self.classes = classes
        self.pre_threshold = pre_threshold  # amplitude threshold to apply before NMS
        self.first_class = first_class

        assert self.first_class >= 0

    def forward(self, bbs, conf):
        """NMS filter confidences based on bounding boxes
        :param bbs: bounding boxes
        :param conf: confidences to filter
        """
        return NMSFilterFunction.apply(
            bbs, conf,
            self.nms_threshold, self.classes, self.pre_threshold, self.first_class
        )

    def extra_repr(self):
        """Extra information
        """
        return 'nms_threshold={}, classes={}{}{}'.format(
            self.nms_threshold, self.classes,
            ", pre_threshold={}".format(self.pre_threshold) if self.pre_threshold > 0 else "",
            ", first_class={}".format(self.first_class) if self.first_class > 0 else "",
        )


# TODO: make these a proper unit test
if __name__ == '__main__':
    n = 4
    a = 3
    c = 10
    net = NMSFilter(classes=c, pre_threshold=0.1)

    bb = torch.empty(n, a, 4)
    xy = bb[:, :, :2]
    wh = bb[:, :, 2:]
    xy.uniform_(0.25, 0.35)  # cluster them within 0.1
    wh.uniform_(0.1, 0.5)

    prob = torch.empty(n, c, a).uniform_(0, .3)
    b = net(bb, prob)

    # now test with cuda
    net = net.cuda()
    bb = bb.cuda()
    prob = prob.cuda()
    b2 = net(bb, prob)

    print((b2.cpu() - b).sum())
