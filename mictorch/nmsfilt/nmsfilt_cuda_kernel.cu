// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include "caffe_cuda.h"
#include "region_common.hpp"

namespace {

template <typename scalar_t>
__device__ void bottom_up_argmerge(const scalar_t* __restrict__ p,
                                   int left, int right, int end,
                                   const int* __restrict__ src, int* __restrict__ dst) {
    int i = left;
    int j = right;
    // Merge 2 already sorted lists
    for (int k = left; k < end; ++k) {
        if (i < right && (j >= end || p[src[i]] > p[src[j]])) {
            dst[k] = src[i];
            i++;
        } else {
            dst[k] = src[j];
            j++;
        }
    }
}

template <typename scalar_t>
__global__ void kernel_channel_argmergesort(
    int outer_num, int channels, int inner_num, int classes, int first_class,
    int width, int chunks,
    const scalar_t* __restrict__ data,
    int* __restrict__ src, int* __restrict__ dst) {
    CUDA_KERNEL_LOOP(index, outer_num * classes * chunks) {
        const int i = index % chunks;
        const int c_idx = (index / chunks) % classes;
        const int c = c_idx + first_class;
        const int n = (index / chunks) / classes;
        const int dim = (n * channels + c) * inner_num;
        const int idx_dim = (n * classes + c_idx) * inner_num;
        int left = i * width;
        int right = min(left + width / 2, inner_num);
        int end = min(left + width, inner_num);
        int* src_idx = src + idx_dim;
        int* dst_idx = dst + idx_dim;
        if (width == 2) {
            // Initialize the index
            if (right < end)
                src_idx[right] = left + 1;
            src_idx[left] = left + 0;
        }
        bottom_up_argmerge(data + dim,
                           left, right, end,
                           src_idx, dst_idx);
    }
}

template <typename scalar_t>
__global__ void kernel_pre_filter(
    int outer_num, int channels, int inner_num, int classes, int first_class,
    float thresh,
    scalar_t* __restrict__ top_conf_data) {
    CUDA_KERNEL_LOOP(index, outer_num * classes * inner_num) {
        const int s = index % inner_num;
        const int c = (index / inner_num) % classes + first_class;
        const int n = (index / inner_num) / classes;
        int dim = (n * channels + c) * inner_num + s;
        if (top_conf_data[dim] <= thresh)
            top_conf_data[dim] = 0;
    }
}

template <typename scalar_t>
__global__ void kernel_nms_filter(
    int outer_num, int channels, int inner_num, int classes, int first_class, int max_output_boxes,
    const int* __restrict__ idx,
    const scalar_t* __restrict__ bbs_data, float thresh,
    scalar_t* __restrict__ top_conf_data) {
    CUDA_KERNEL_LOOP(index, outer_num * classes) {
        const int c_idx = index % classes;
        const int c = c_idx + first_class;
        const int n = index / classes;
        const int dim = (n * channels + c) * inner_num;
        const int idx_dim = (n * classes + c_idx) * inner_num;
        const int* src_idx = idx + idx_dim;
        int non_zero_count = 0;
        for (int i_idx = 0; i_idx < inner_num; ++i_idx) {
            int i = src_idx[i_idx];
            if (top_conf_data[dim + i] == 0)
                continue;
            if (non_zero_count == max_output_boxes) {
                // zero out the rest
                top_conf_data[dim + i] = 0;
                continue;
            }
            ++non_zero_count;
            auto i_bb = bbs_data + (n * inner_num + i) * 4;
            for (int j_idx = i_idx + 1; j_idx < inner_num; ++j_idx) {
                int j = src_idx[j_idx];
                if (top_conf_data[dim + j] == 0)
                    continue;
                auto j_bb = bbs_data + (n * inner_num + j) * 4;
                scalar_t curr_iou = TBoxIou<scalar_t>(i_bb[0], i_bb[1], i_bb[2], i_bb[3],
                                                      j_bb[0], j_bb[1], j_bb[2], j_bb[3]);
                if (curr_iou > thresh)
                    top_conf_data[dim + j] = 0;
            }
        }
    }
}


} // namespace

std::vector<at::Tensor> nmsfilt_cuda_forward(
    at::Tensor bbs, at::Tensor conf,
    float nms_threshold, int classes, float pre_threshold, int first_class, int max_output_boxes,
    int outer_num, int channels, int inner_num) {

  auto top_conf = conf.clone();

  if (pre_threshold >= 0) {
    AT_DISPATCH_FLOATING_TYPES(conf.scalar_type(), "nmsfilt_cuda_forward::kernel_pre_filter", ([&] {
      kernel_pre_filter<scalar_t><<<GET_BLOCKS(outer_num * classes * inner_num), CUDA_NUM_THREADS>>>(
          outer_num, channels, inner_num, classes, first_class,
          pre_threshold,
          top_conf.data<scalar_t>());
    }));
  }

  if (nms_threshold <= 0 || inner_num == 1)
    return {top_conf};

  // intermediate variables
  auto idx = at::empty({outer_num, classes, inner_num}, bbs.options().dtype(at::kInt));
  int* idx_data = idx.data<int>();

  {
    // This memory is safe to release after sorting but we keep it in GPU memory,
    auto idx_swp = at::empty_like(idx);
    int* idx_tmp = idx_swp.data<int>();
    // Start swapped if loop runs for an odd number
    bool is_swapped = ((int)ceil(log2((double)inner_num))) % 2 != 0;
    AT_DISPATCH_FLOATING_TYPES(conf.scalar_type(), "nmsfilt_cuda_forward::kernel_channel_argmergesort", ([&] {
      for (int width = 2; width < inner_num * 2; width *= 2) {
        int chunks = (inner_num + width - 1) / width;
        int* src_idx = is_swapped ? idx_tmp : idx_data;
        int* dst_idx = is_swapped ? idx_data : idx_tmp;
        kernel_channel_argmergesort<scalar_t><<<GET_BLOCKS(outer_num * classes * chunks), CUDA_NUM_THREADS>>>(
            outer_num, channels, inner_num, classes, first_class,
            width, chunks,
            conf.data<scalar_t>(),
            src_idx, dst_idx);
        is_swapped = !is_swapped;
      }
    }));
  }

  AT_DISPATCH_FLOATING_TYPES(conf.scalar_type(), "nmsfilt_cuda_forward::kernel_nms_filter", ([&] {
    kernel_nms_filter <<<GET_BLOCKS(outer_num * classes), CUDA_NUM_THREADS >>>(
        outer_num, channels, inner_num, classes, first_class, max_output_boxes,
        idx.data<int>(),
        bbs.data<scalar_t>(), nms_threshold,
        top_conf.data<scalar_t>()
        );
  }));

  return {top_conf};
}
