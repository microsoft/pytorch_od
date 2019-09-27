# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.autograd import Function

try:
    import nmsfilt_cuda
except ImportError:
    nmsfilt_cuda = None
import nmsfilt_cpu


class NMSFilterFunction(Function):
    @staticmethod
    def forward(ctx, bbs, conf,
                nms_threshold, classes, pre_threshold, first_class, max_output_boxes):
        if bbs.is_cuda:
            nms_ = nmsfilt_cuda
        else:
            nms_ = nmsfilt_cpu
        top_conf = nms_.forward(bbs, conf,
                                nms_threshold, classes, pre_threshold, first_class, max_output_boxes)[0]

        return top_conf

    @staticmethod
    def backward(ctx, grad_output):
        return tuple([None] * 6)


class NMSFilter(nn.Module):
    """Applies Non-Maximal Suppression filter to bounding box confidence values
    Each class (and each batch) will be filtered independently
    """
    def __init__(self, nms_threshold=0.45, classes=1, pre_threshold=-1.0, first_class=0, max_output_boxes=-1,
                 return_bbs=False):
        """NMSFilter
        :param nms_threshold: NMS threshold
        :param classes: number of classes to filter (-1 to filter all classes independently)
        :param pre_threshold: amplitude threshold to apply before NMS
        :param first_class: The first class to start filtering "classes"
        :param max_output_boxes: maximum number of boxes per-class per-batch (<= 0 to ignore)
        :param return_bbs: if should return the input bbs as well as filtered conf
        """
        super(NMSFilter, self).__init__()
        self.nms_threshold = nms_threshold
        self.classes = classes
        self.pre_threshold = pre_threshold  # amplitude threshold to apply before NMS
        self.first_class = first_class
        self.max_output_boxes = max_output_boxes
        self.return_bbs = return_bbs

        assert self.first_class >= 0

    def forward(self, bbs, conf):
        """NMS filter confidences based on bounding boxes
        :param bbs: bounding boxes
        :param conf: confidences to filter
        """
        if isinstance(bbs.shape[0], torch.Tensor):
            assert self.classes <= 1, "multi-class NMS tracing not supported yet"
            filt = torch.ops.mtorch_ops.nmsfilt(
                bbs, conf,
                self.nms_threshold, self.pre_threshold, self.max_output_boxes
            )
        else:
            filt = NMSFilterFunction.apply(
                bbs, conf,
                self.nms_threshold, self.classes, self.pre_threshold, self.first_class, self.max_output_boxes
            )

        if self.return_bbs:
            return bbs, filt
        return filt

    def extra_repr(self):
        """Extra information
        """
        return 'nms_threshold={}, classes={}{}{}{}'.format(
            self.nms_threshold, self.classes,
            ", pre_threshold={}".format(self.pre_threshold) if self.pre_threshold > 0 else "",
            ", first_class={}".format(self.first_class) if self.first_class > 0 else "",
            ", max_output_boxes={}".format(self.max_output_boxes) if self.max_output_boxes > 0 else "",
        )


def register_custom_nms_op():
    # experimenting custom op registration.
    from torch.onnx.symbolic_helper import parse_args
    from torch.onnx.symbolic_opset9 import view, select, index_select, scatter
    @parse_args('v', 'v', 'f', 'f', 'i')
    def symbolic_nmsfilt(g, boxes, scores, iou_threshold, score_threshold, max_output_boxes):
        # if should return all
        if max_output_boxes <= 0:
            max_output_boxes = 10000
        shape = g.op("Shape", scores)  # original shape
        boxes = view(g, boxes, (1, -1, 4))
        max_output_per_class = g.op('Constant', value_t=torch.tensor([max_output_boxes], dtype=torch.long))
        iou_threshold = g.op('Constant', value_t=torch.tensor([iou_threshold], dtype=torch.float))
        score_threshold = g.op('Constant', value_t=torch.tensor([score_threshold], dtype=torch.float))
        # center_point_box == 1 is for our center_x, centr_y, width, height format
        nms_out = g.op('NonMaxSuppression',
                       boxes, view(g, scores, (1, 1, -1)), max_output_per_class, iou_threshold, score_threshold,
                       center_point_box_i=1)
        idx = view(g, select(g, nms_out, 1, g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))), (-1,))
        scores = view(g, scores, (-1,))
        flat_shape = g.op("Shape", scores)
        src = index_select(g, scores, 0, idx)
        src = view(g, src, (-1,))
        filt = g.op("ConstantOfShape", flat_shape)
        filt = scatter(g, filt, 0, idx, src)
        return view(g, filt, shape)

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('mtorch_ops::nmsfilt', symbolic_nmsfilt, 10)


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
