# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
import nms_cpu

nms = torch.ops.mtorch_ops.nms


def register_custom_nms_op():
    # experimenting custom op registration.
    from torch.onnx.symbolic_helper import parse_args
    from torch.onnx.symbolic_opset9 import view, select
    @parse_args('v', 'v', 'f', 'i')
    def symbolic_nms(g, boxes, scores, iou_threshold, max_output_boxes):
        # if should return all
        if max_output_boxes <= 0:
            max_output_boxes = 10000
        boxes = view(g, boxes, (1, -1, 4))
        max_output_per_class = g.op('Constant', value_t=torch.tensor([max_output_boxes], dtype=torch.long))
        iou_threshold = g.op('Constant', value_t=torch.tensor([iou_threshold], dtype=torch.float))
        # center_point_box == 1 is for our center_x, centr_y, width, height format
        nms_out = g.op('NonMaxSuppression',
                       boxes, view(g, scores, (1, 1, -1)), max_output_per_class, iou_threshold,
                       center_point_box_i=1)
        idx = select(g, nms_out, 1, g.op('Constant', value_t=torch.tensor([2], dtype=torch.long)))
        return view(g, idx, (-1,))

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('mtorch_ops::nms', symbolic_nms, 10)
