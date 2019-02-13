// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/torch.h>

#include <vector>

#include "mtorch_common.h"

// CUDA forward declarations

std::vector<at::Tensor> smtpred_cuda_forward(
    at::Tensor conf, at::Tensor obj,
    at::Tensor group_offset, at::Tensor group_size, at::Tensor child, at::Tensor child_size,
    float threshold, bool output_tree_path, bool append_max,
    int root_size, int stack_size,
    int outer_num, int inner_num
);

// C++ interface
std::vector<at::Tensor> smtpred_forward(
    at::Tensor conf, at::Tensor obj,
    at::Tensor group_offset, at::Tensor group_size, at::Tensor child, at::Tensor child_size,
    float threshold, bool output_tree_path, bool append_max,
    int root_size, int stack_size
) {
  CHECK_INPUT(conf);
  CHECK_INPUT(obj);
  CHECK_INPUT(group_offset);
  CHECK_INPUT(group_size);
  CHECK_INPUT(child);
  CHECK_INPUT(child_size);

  AT_ASSERTM(conf.dim() >= 2, "invalid conf dim");

  int outer_num = conf.size(0);
  int inner_num = 1;
  for (int i = 2; i < conf.dim(); ++i)
    inner_num *= conf.size(i);

  return smtpred_cuda_forward(
      conf, obj,
      group_offset, group_size, child, child_size,
      threshold, output_tree_path, append_max,
      root_size, stack_size,
      outer_num, inner_num);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &smtpred_forward, "SoftmaxTreePrediction forward (CUDA)");
}
