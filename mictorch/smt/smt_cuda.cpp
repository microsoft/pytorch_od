// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/torch.h>

#include <vector>

#include "mtorch_common.h"

// CUDA forward declarations

std::vector<at::Tensor> smt_cuda_forward(
    at::Tensor input,
    at::Tensor group_offset, at::Tensor group_size,
    int outer_num, int inner_num, int axis);

std::vector<at::Tensor> smt_cuda_backward(
    at::Tensor prob, at::Tensor grad_output,
    at::Tensor group_offset, at::Tensor group_size,
    int outer_num, int inner_num, int axis);

// C++ interface
std::vector<at::Tensor> smt_forward(
    at::Tensor input,
    at::Tensor group_offset, at::Tensor group_size,
    int axis) {
  CHECK_INPUT(input);
  CHECK_INPUT(group_offset);
  CHECK_INPUT(group_size);

  int outer_num = 1;
  for (int i = 0; i < axis; ++i)
    outer_num *= input.size(i);
  int inner_num = 1;
  for (int i = axis + 1; i < input.dim(); ++i)
    inner_num *= input.size(i);

  return smt_cuda_forward(input,
      group_offset, group_size,
      outer_num, inner_num,
      axis);
}

std::vector<at::Tensor> smt_backward(
    at::Tensor prob, at::Tensor grad_output,
    at::Tensor group_offset, at::Tensor group_size,
    int axis) {
  CHECK_INPUT(prob);
  CHECK_INPUT(grad_output);
  CHECK_INPUT(group_offset);
  CHECK_INPUT(group_size);

  int outer_num = 1;
  for (int i = 0; i < axis; ++i)
    outer_num *= prob.size(i);
  int inner_num = 1;
  for (int i = axis + 1; i < prob.dim(); ++i)
    inner_num *= prob.size(i);

  return smt_cuda_backward(
      prob, grad_output,
      group_offset, group_size,
      outer_num, inner_num,
      axis);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &smt_forward, "SMT forward (CUDA)");
  m.def("backward", &smt_backward, "SMT backward (CUDA)");
}
