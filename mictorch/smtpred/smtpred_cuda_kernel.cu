// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cfloat>

#include "caffe_cuda.h"
#include "region_common.hpp"

namespace {

__device__ void stack_push(double* __restrict__ parent_p_data, int* __restrict__ parent_argmax_data, int* __restrict__ g_data,
                           int& stack_size,
                           double p, int argmax, int g) {
    parent_p_data[stack_size] = p;
    parent_argmax_data[stack_size] = argmax;
    g_data[stack_size] = g;
    stack_size++;
}

__device__ void stack_pop(const double* __restrict__ parent_p_data, const int* __restrict__ parent_argmax_data, const int* __restrict__ g_data,
                          int& stack_size,
                          double& p, int& argmax, int& g) {
    assert(stack_size > 0);
    stack_size--;
    p = parent_p_data[stack_size];
    argmax = parent_argmax_data[stack_size];
    g = g_data[stack_size];
}

template <typename scalar_t>
__device__ void predict_tree_stack(
    int outer_num, int channels, int inner_num,
    bool append_max,
    float threshold,
    const int* __restrict__ group_offset_data, const int* __restrict__ group_size_data, const int* __restrict__ child_data, const int* __restrict__ child_size_data,
    double* __restrict__ parent_p_data, int* __restrict__ parent_argmax_data, int* __restrict__ g_data,
    const scalar_t* __restrict__ obj_data, const scalar_t* __restrict__ prob_data,
    int max_stack_size, int n, int s, int g,
    scalar_t* top_data,
    bool output_tree_path) {

    int stack_size = 0;
    const int top_channels = append_max ? (channels + 1) : channels;
    scalar_t obj = obj_data ? obj_data[n * inner_num + s] : 1;
    double root_p = output_tree_path ? obj : 1.0;
    threshold = output_tree_path ? (threshold * obj) : threshold;
    stack_push(parent_p_data, parent_argmax_data, g_data,
               stack_size,
               root_p, -1, g);
    while (stack_size) {
        assert(stack_size <= max_stack_size);
        double parent_p;
        int parent_argmax;
        int g;
        stack_pop(parent_p_data, parent_argmax_data, g_data,
                  stack_size,
                  parent_p, parent_argmax, g);
        double p = parent_p;
        int argmax = 0;
        {
            scalar_t maxval = -FLT_MAX;
            auto offset = group_offset_data[g];
            argmax = offset;
            auto size = group_size_data[g];
            for (int j = 0; j < size; ++j) {
                scalar_t prob = prob_data[(n * channels + offset + j) * inner_num + s];
                if (prob > maxval) {
                    argmax = offset + j;
                    maxval = prob;
                }
            }
            p *= maxval;
        }
        if (p > threshold) {
            if (output_tree_path) {
                top_data[(n * top_channels + argmax) * inner_num + s] = static_cast<scalar_t>(p);
            }
            g = child_data[argmax]; // initial child group
            if (g >= 0) {
                // if there is any child, descend further
                int sg_count = child_size_data[argmax] + 1;
                for (int sg = 0; sg < sg_count; ++sg) {
                    stack_push(parent_p_data, parent_argmax_data, g_data,
                               stack_size,
                               p, argmax, g + sg);

                }
                continue;
            }
        } else {
            argmax = parent_argmax;
            if (argmax < 0)
                continue;
            p = parent_p;
        }
        
        scalar_t node_p = 0;
        if (!output_tree_path) {
            node_p = obj_data ? obj : static_cast<scalar_t>(p);
            top_data[(n * top_channels + argmax) * inner_num + s] = node_p;
        }
        if (append_max) {
            int max_idx = (n * top_channels + channels) * inner_num + s;
            if (output_tree_path) {
                // in this case, we use the obj as the max value, which will be
                // used as the indicator for class-independent NMS. or the
                // maximum value will always be the ones in the root.
                // gradually, we might remove the support of append_max since
                // it is more like a legacy strategy
                top_data[max_idx] = obj;
            } else {
                if (node_p > top_data[max_idx]) {
                    top_data[max_idx] = node_p;
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void kernel_smt_prediction(
    int outer_num, int channels, int inner_num, int root_size,
    bool append_max,
    float threshold,
    const int* __restrict__ group_offset_data, const int* __restrict__ group_size_data, const int* __restrict__ child_data, const int* __restrict__ child_size_data,
    double* __restrict__ parent_p_data, int* __restrict__ parent_argmax_data, int* __restrict__ g_data,
    const scalar_t* __restrict__ obj_data, const scalar_t* __restrict__ prob_data,
    int max_stack_size,
    scalar_t* __restrict__ top_data,
    bool output_tree_path) {
    CUDA_KERNEL_LOOP(index, outer_num * root_size * inner_num) {
        const int s = index % inner_num;
        const int g = (index / inner_num) % root_size;
        const int n = (index / inner_num) / root_size;

        predict_tree_stack(outer_num, channels, inner_num,
                           append_max,
                           threshold,
                           group_offset_data, group_size_data, child_data, child_size_data,
                           &parent_p_data[index * max_stack_size], &parent_argmax_data[index * max_stack_size], &g_data[index * max_stack_size],
                           obj_data, prob_data,
                           max_stack_size, n, s, g,
                           top_data, 
                           output_tree_path);
    }
}


} // namespace

std::vector<at::Tensor> smtpred_cuda_forward(
    at::Tensor conf, at::Tensor obj,
    at::Tensor group_offset, at::Tensor group_size, at::Tensor child, at::Tensor child_size,
    float threshold, bool output_tree_path, bool append_max,
    int root_size, int stack_size,
    int outer_num, int inner_num
) {

  root_size++;

  // Intermediate variables
  auto stack_parent_p = at::empty({outer_num, root_size, inner_num, stack_size}, at::CUDA(at::kDouble));
  auto stack_parent_argmax = at::empty({outer_num, root_size, inner_num, stack_size}, at::CUDA(at::kInt));
  auto stack_g = at::empty({outer_num, root_size, inner_num, stack_size}, at::CUDA(at::kInt));

  auto shape = conf.sizes().vec();
  int channels = shape[1];
  if (append_max)
    shape[1] = channels + 1;

  auto top = at::zeros(shape, conf.type());

  AT_DISPATCH_FLOATING_TYPES(conf.type(), "smtpred_cuda_forward::kernel_smt_prediction", ([&] {
    scalar_t* obj_data = nullptr;
    if (obj.numel())
      obj_data = obj.data<scalar_t>();
    kernel_smt_prediction<scalar_t><<<GET_BLOCKS(outer_num * root_size * inner_num), CUDA_NUM_THREADS>>>(
        outer_num, channels, inner_num, root_size,
        append_max,
        threshold,
        group_offset.data<int>(), group_size.data<int>(), child.data<int>(), child_size.data<int>(),
        stack_parent_p.data<double>(), stack_parent_argmax.data<int>(), stack_g.data<int>(),
        obj_data, conf.data<scalar_t>(),
        stack_size,
        top.data<scalar_t>(),
        output_tree_path);
  }));

  return {top};
}
