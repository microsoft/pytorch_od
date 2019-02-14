// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cfloat>
#include "caffe_cuda.h"

namespace {

template <typename scalar_t>
__global__ void kernel_subtract_max(const int num, const int channels, const int spatial_dim, const int groups,
                                    const int* __restrict__ group_offset_data, const int* __restrict__ group_size_data, scalar_t* __restrict__ data) {
    CUDA_KERNEL_LOOP(index, num * groups * spatial_dim) {
        int s = index % spatial_dim;
        int g = (index / spatial_dim) % groups;
        int n = (index / spatial_dim) / groups;
        auto offset = group_offset_data[g];
        auto size = group_size_data[g];
        scalar_t maxval = -FLT_MAX;
        for (int j = 0; j < size; ++j) {
            if (data[(n * channels + offset + j) * spatial_dim + s] > maxval)
                maxval = data[(n * channels + offset + j) * spatial_dim + s];
        }
        // TODO: Use dynamic parallelism for devices with 3.5 compute capability
        // Subtract the max
        for (int j = 0; j < size; ++j)
            data[(n * channels + offset + j) * spatial_dim + s] -= maxval;
    }
}

template <typename scalar_t>
__global__ void kernel_div_sum(const int num, const int channels, const int spatial_dim, const int groups,
                               const int* __restrict__ group_offset_data, const int* __restrict__ group_size_data, scalar_t* __restrict__ data) {
    CUDA_KERNEL_LOOP(index, num * groups * spatial_dim) {
        int s = index % spatial_dim;
        int g = (index / spatial_dim) % groups;
        int n = (index / spatial_dim) / groups;
        auto offset = group_offset_data[g];
        auto size = group_size_data[g];
        scalar_t sum = 0;
        for (int j = 0; j < size; ++j)
            sum += data[(n * channels + offset + j) * spatial_dim + s];
        // TODO: Use dynamic parallelism for devices with 3.5 compute capability
        // divide by sum
        for (int j = 0; j < size; ++j)
            data[(n * channels + offset + j) * spatial_dim + s] /= sum;
    }
}

template <typename scalar_t>
__global__ void kernel_subtract_dot(const int num, const int channels, const int spatial_dim, const int groups,
                                    const int* group_offset_data, const int* group_size_data, 
                                    const scalar_t* __restrict__ data_1, const scalar_t* __restrict__ data_2, scalar_t* __restrict__ out) {
    CUDA_KERNEL_LOOP(index, num * groups * spatial_dim) {
        int s = index % spatial_dim;
        int g = (index / spatial_dim) % groups;
        int n = (index / spatial_dim) / groups;
        auto offset = group_offset_data[g];
        auto size = group_size_data[g];
        scalar_t dot = 0;
        for (int j = 0; j < size; ++j) {
            dot += (data_1[(n * channels + offset + j) * spatial_dim + s]
                    * data_2[(n * channels + offset + j) * spatial_dim + s]);
        }
        // TODO: Use dynamic parallelism for devices with 3.5 compute capability
        // subtract the dot
        for (int j = 0; j < size; ++j)
            out[(n * channels + offset + j) * spatial_dim + s] -= dot;
    }
}

} // namespace

std::vector<at::Tensor> smt_cuda_forward(
    at::Tensor input,
    at::Tensor group_offset, at::Tensor group_size,
    int outer_num, int inner_num, int axis) {

  int groups = group_offset.numel();
  int channels = input.size(axis);

  auto prob = input.clone();

  // We need to subtract the per-group max to avoid numerical issues, compute the exp,
  // and then per-group normalize.
  AT_DISPATCH_FLOATING_TYPES(input.type(), "smt_cuda_forward", ([&] {
    kernel_subtract_max<scalar_t><<<GET_BLOCKS(outer_num * groups * inner_num), CUDA_NUM_THREADS>>>(
        outer_num, channels, inner_num, groups,
        group_offset.data<int>(), group_size.data<int>(),
        prob.data<scalar_t>());
  }));

  // exponentiate
  prob.exp_();

  // per-group sum after exp, and divide
  AT_DISPATCH_FLOATING_TYPES(input.type(), "smt_cuda_forward", ([&] {
    kernel_div_sum<scalar_t><<<GET_BLOCKS(outer_num * groups * inner_num), CUDA_NUM_THREADS>>>(
        outer_num, channels, inner_num, groups,
        group_offset.data<int>(), group_size.data<int>(),
        prob.data<scalar_t>());
  }));

  return {prob};
}

std::vector<at::Tensor> smt_cuda_backward(
    at::Tensor prob, at::Tensor grad_output,
    at::Tensor group_offset, at::Tensor group_size,
    int outer_num, int inner_num, int axis) {

  int groups = group_offset.numel();
  int channels = prob.size(axis);

  auto diff = grad_output.clone(); // bottom diff

  // Compute per-group inner1d(top_diff, top_data) and subtract them from the bottom diff.
  AT_DISPATCH_FLOATING_TYPES(prob.type(), "smt_cuda_backward", ([&] {
    kernel_subtract_dot<scalar_t><<<GET_BLOCKS(outer_num * groups * inner_num), CUDA_NUM_THREADS>>>(
        outer_num, channels, inner_num, groups,
        group_offset.data<int>(), group_size.data<int>(),
        grad_output.data<scalar_t>(), prob.data<scalar_t>(), diff.data<scalar_t>());
  }));

  // elementwise multiplication
  diff.mul_(prob);

  return {diff};
}
