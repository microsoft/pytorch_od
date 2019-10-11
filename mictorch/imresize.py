# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch


@torch.jit.script
def _dynsize_helper(crop_height_i, crop_width_i):
    """The input shape could be dynamic
    This will be exported as .ones().nonzero() with proper params
    """
    y = torch.arange(crop_height_i, dtype=torch.float32)
    x = torch.arange(crop_width_i, dtype=torch.float32)
    return y, x


def resize_bilinear(im,
                    resized_shape=None, output_crop_shape=None,
                    darknet=False, edge=True, axis=2):
    """Bilinear interpolate
    :param im: Image tensor shape (1xCxHxW)
    :type im: torch.Tensor
    :param resized_shape: shape of the resized image (H_r, W_r)
    :param output_crop_shape: shape of the output center crop (H_c, W_c)
    :param darknet: if should resize darknet-style
    :param edge: if should use edge (like in OpenCV)
    :param axis: height axis (0 or 2)
    :return: resized image
    :rtype: torch.Tensor
    """

    if resized_shape is None:
        assert output_crop_shape is not None, "No dimension given to resize"
        resized_shape = output_crop_shape

    input_height, input_width = im.shape[axis:axis + 2]
    if not isinstance(input_height, torch.Tensor):
        input_height, input_width = torch.tensor(input_height), torch.tensor(input_width)
    input_height, input_width = input_height.float(), input_width.float()

    assert resized_shape is not None, "No dimension given to resize"
    target_height, target_width = resized_shape
    if not isinstance(target_height, torch.Tensor):
        target_height, target_width = torch.tensor(target_height), torch.tensor(target_width)
    resized_shape_i = target_height, target_width
    target_height, target_width = target_height.float(), target_width.float()
    resized_shape = target_height, target_width

    top = left = None
    if output_crop_shape is None:
        crop_height_i, crop_width_i = resized_shape_i
        crop_height, crop_width = resized_shape
        top = 0
        left = 0
    else:
        crop_height_i, crop_width_i = output_crop_shape
        if not isinstance(crop_height_i, torch.Tensor):
            crop_height_i, crop_width_i = torch.tensor(crop_height_i), torch.tensor(crop_width_i)
        crop_height, crop_width = crop_height_i, crop_width_i

    if not crop_height.dtype.is_floating_point:
        crop_height, crop_width = crop_height.float(), crop_width.float()

    # TODO: ONNX does not like float in arange, can avoid .long() once issue #27718 is fixed in release
    if crop_height_i.dtype.is_floating_point:
        crop_height_i, crop_width_i = crop_height_i.long(), crop_width_i.long()

    # TODO: Use normal arange once issue #20075 is fixed in release
    y, x = _dynsize_helper(crop_height_i, crop_width_i)
    y, x = y.to(im.device), x.to(im.device)

    if top is None:
        assert left is None
        assert crop_height <= target_height and crop_width <= target_width, "invalid output_crop_shape"
        if not crop_height.dtype.is_floating_point:
            crop_height, crop_width = crop_height.float(), crop_width.float()
        # TODO: use .round() when PyTorch Issue # 25806 is fixed (round for ONNX is released)
        top = ((target_height - crop_height) / 2 + 0.5).floor()
        left = ((target_width - crop_width) / 2 + 0.5).floor()

    rh = target_height / input_height
    rw = target_width / input_width
    if edge:
        ty = (y + top + 1) / rh + 0.5 * (1 - 1.0 / rh) - 1
        tx = (x + left + 1) / rw + 0.5 * (1 - 1.0 / rw) - 1
        zero = torch.tensor(0.0, dtype=torch.float32)
        ty = torch.max(ty, zero)  # ty[ty < 0] = 0
        tx = torch.max(tx, zero)  # tx[tx < 0] = 0
    else:
        ty = (y + top) / rh
        tx = (x + left) / rw
    del y, x

    ity0 = ty.floor()
    if darknet:
        ity1 = ity0 + 1
    else:
        ity1 = ty.ceil()

    itx0 = tx.floor()
    if darknet:
        itx1 = itx0 + 1
    else:
        itx1 = tx.ceil()

    dy = ty - ity0
    dx = tx - itx0
    del ty, tx
    if axis == 0:
        dy = dy.view(-1, 1, 1)
        dx = dx.view(-1, 1)
    else:
        assert axis == 2, "Only 1xCxHxW and HxWxC inputs supported"
        dy = dy.view(-1, 1)
        dx = dx.view(-1)
    dydx = dy * dx

    # noinspection PyProtectedMember
    if torch._C._get_tracing_state():
        # always do clamp when tracing
        ity1 = torch.min(ity1, input_height - 1)
        itx1 = torch.min(itx1, input_width - 1)
    else:
        # TODO: use searchsorted once avaialble
        # items at the end could be out of bound (if upsampling)
        if ity1[-1] >= input_height:
            ity1[ity1 >= input_height] = input_height - 1
        if itx1[-1] >= input_width:
            itx1[itx1 >= input_width] = input_width - 1

    iy0 = ity0.long()
    ix0 = itx0.long()
    iy1 = ity1.long()
    ix1 = itx1.long()
    del ity0, itx0, ity1, itx1

    if not im.dtype.is_floating_point:
        im = im.float()
    im_iy0 = im.index_select(axis, iy0)
    im_iy1 = im.index_select(axis, iy1)
    d = im_iy0.index_select(axis + 1, ix0) * (1 - dx - dy + dydx) + \
        im_iy1.index_select(axis + 1, ix0) * (dy - dydx) + \
        im_iy0.index_select(axis + 1, ix1) * (dx - dydx) + \
        im_iy1.index_select(axis + 1, ix1) * dydx

    return d
