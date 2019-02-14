// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/torch.h>

#include <vector>

#include "mtorch_common.h"

// CUDA forward declarations

std::vector<at::Tensor> nmsfilt_cuda_forward(
    at::Tensor bbs, at::Tensor conf,
    float nms_threshold, int classes, float pre_threshold, int first_class,
    int outer_num, int channels, int inner_num);

// C++ interface
std::vector<at::Tensor> nmsfilt_forward(
    at::Tensor bbs, at::Tensor conf,
    float nms_threshold, int classes, float pre_threshold, int first_class) {
  CHECK_INPUT(bbs);
  CHECK_INPUT(conf);

  AT_ASSERTM(bbs.dim() >= 2, "invalid bbs dim");
  AT_ASSERTM(conf.dim() >= 2, "invalid conf dim");
  int bbs_axis = bbs.dim() - 1;  // Last axis
  AT_ASSERTM(bbs.size(bbs_axis) == 4, "bbs axis must have 4 corners");

  int outer_num = bbs.size(0);
  AT_ASSERTM(conf.size(0) == outer_num, "conf has invalid number of batches");
  int inner_num = 1;
  for (int i = 1; i < bbs_axis; ++i)
    inner_num *= bbs.size(i);
  AT_ASSERTM(bbs.numel() == inner_num * outer_num * 4, "bbs invalid size");

  int channels = 1;
  if (conf.numel() != inner_num * outer_num)
    channels = conf.size(1);
  AT_ASSERTM(classes <= channels, "classes must be less than channels");
  AT_ASSERTM(conf.numel(), inner_num * channels * outer_num, "conf invalid size");

  if (classes <= 0)
    classes = channels;
  AT_ASSERTM(classes + first_class <= channels, "classes + first_class_ must be <= channels");

  return nmsfilt_cuda_forward(
      bbs, conf,
      nms_threshold, classes, pre_threshold, first_class,
      outer_num, channels, inner_num);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &nmsfilt_forward, "NMSFilter forward (CUDA)");
}
