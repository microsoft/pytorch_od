// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/torch.h>

#include <vector>
#include <cfloat>

#include "mtorch_common.h"


// C++ interface
std::vector<at::Tensor> smt_forward(
    at::Tensor input,
    at::Tensor group_offset, at::Tensor group_size,
    int axis) {
  CHECK_INPUT_CPU(input);
  CHECK_INPUT_CPU(group_offset);
  CHECK_INPUT_CPU(group_size);

  int outer_num = 1;
  for (int i = 0; i < axis; ++i)
    outer_num *= input.size(i);
  int inner_num = 1;
  for (int i = axis + 1; i < input.dim(); ++i)
    inner_num *= input.size(i);

  int groups = group_offset.numel();
  int channels = input.size(axis);

  auto prob = input.clone();

  // We need to subtract the per-group max to avoid numerical issues, compute the exp,
  // and then per-group normalize.
  AT_DISPATCH_FLOATING_TYPES(input.type(), "smt_cpu_forward", ([&] {
    const int* group_offset_data = group_offset.data<int>();
    const int* group_size_data = group_size.data<int>();
    scalar_t* data = prob.data<scalar_t>();
    for (int index = 0; index < outer_num * groups * inner_num; ++index) {
      int s = index % inner_num;
      int g = (index / inner_num) % groups;
      int n = (index / inner_num) / groups;
      auto offset = group_offset_data[g];
      auto size = group_size_data[g];
      scalar_t maxval = -FLT_MAX;
      for (int j = 0; j < size; ++j) {
        if (data[(n * channels + offset + j) * inner_num + s] > maxval)
          maxval = data[(n * channels + offset + j) * inner_num + s];
      }
      // Subtract the max
      for (int j = 0; j < size; ++j)
        data[(n * channels + offset + j) * inner_num + s] -= maxval;
    }
  }));

  // exponentiate
  prob.exp_();

  // per-group sum after exp, and divide
  AT_DISPATCH_FLOATING_TYPES(input.type(), "smt_cpu_forward", ([&] {
    const int* group_offset_data = group_offset.data<int>();
    const int* group_size_data = group_size.data<int>();
    scalar_t* data = prob.data<scalar_t>();
    for (int index = 0; index < outer_num * groups * inner_num; ++index) {
      int s = index % inner_num;
      int g = (index / inner_num) % groups;
      int n = (index / inner_num) / groups;
      auto offset = group_offset_data[g];
      auto size = group_size_data[g];
      scalar_t sum = 0;
      for (int j = 0; j < size; ++j)
        sum += data[(n * channels + offset + j) * inner_num + s];
      // divide by sum
      for (int j = 0; j < size; ++j)
        data[(n * channels + offset + j) * inner_num + s] /= sum;
    }
  }));

  return {prob};
}

std::vector<at::Tensor> smt_backward(
    at::Tensor prob, at::Tensor grad_output,
    at::Tensor group_offset, at::Tensor group_size,
    int axis) {
  CHECK_INPUT_CPU(prob);
  CHECK_INPUT_CPU(grad_output);
  CHECK_INPUT_CPU(group_offset);
  CHECK_INPUT_CPU(group_size);

  int outer_num = 1;
  for (int i = 0; i < axis; ++i)
    outer_num *= prob.size(i);
  int inner_num = 1;
  for (int i = axis + 1; i < prob.dim(); ++i)
    inner_num *= prob.size(i);

  int groups = group_offset.numel();
  int channels = prob.size(axis);

  auto diff = grad_output.clone(); // bottom diff

  // Compute per-group inner1d(top_diff, top_data) and subtract them from the bottom diff.
  AT_DISPATCH_FLOATING_TYPES(prob.type(), "smt_cpu_backward", ([&] {
    const int* group_offset_data = group_offset.data<int>();
    const int* group_size_data = group_size.data<int>();
    const scalar_t* data_1 = grad_output.data<scalar_t>();
    const scalar_t* data_2 = prob.data<scalar_t>();
    scalar_t* out = diff.data<scalar_t>();
    for(int index = 0; index < outer_num * groups * inner_num; ++index) {
      int s = index % inner_num;
      int g = (index / inner_num) % groups;
      int n = (index / inner_num) / groups;
      auto offset = group_offset_data[g];
      auto size = group_size_data[g];
      scalar_t dot = 0;
      for (int j = 0; j < size; ++j) {
        dot += (data_1[(n * channels + offset + j) * inner_num + s]
                * data_2[(n * channels + offset + j) * inner_num + s]);
      }
      // subtract the dot
      for (int j = 0; j < size; ++j)
        out[(n * channels + offset + j) * inner_num + s] -= dot;
    }
  }));

  // elementwise multiplication
  diff.mul_(prob);

  return {diff};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &smt_forward, "SMT forward (CPU)");
  m.def("backward", &smt_backward, "SMT backward (CPU)");
}
