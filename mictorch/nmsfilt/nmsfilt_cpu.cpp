// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <torch/script.h>

#include <vector>
#include <numeric>

#include "mtorch_common.h"
#include "region_common.hpp"

namespace {

// sort the values in p in descending order and keep the index in result
template <typename scalar_t>
void sort_nms_idx(const scalar_t* p,
                  std::vector<int>& result) {
    std::iota(result.begin(), result.end(), 0);
    std::sort(result.begin(), result.end(),
              [p](int i, int j) {
        return p[i] > p[j];
    });
}
template <typename scalar_t>
void pre_filter(int outer_num, int channels, int inner_num, int classes, int first_class,
                float thresh,
                scalar_t* RESTRICT top_conf_data) {
    for (int index = 0; index < outer_num * classes * inner_num; ++index) {
        const int s = index % inner_num;
        const int c = (index / inner_num) % classes + first_class;
        const int n = (index / inner_num) / classes;
        int dim = (n * channels + c) * inner_num + s;
        if (top_conf_data[dim] <= thresh)
            top_conf_data[dim] = 0;
    }
}

template <typename scalar_t>
void nms_filter(const scalar_t* RESTRICT bbs_data,
                int outer_num, int channels, int inner_num, int classes, int first_class, int max_output_boxes,
                float thresh,
                scalar_t* RESTRICT top_conf_data) {

    for (int index = 0; index < outer_num * classes; ++index) {
        int c = index % classes + first_class;
        int n = index / classes;

        const int dim = (n * channels + c) * inner_num;
        std::vector<int> idx(inner_num);
        sort_nms_idx<scalar_t>(top_conf_data + dim, idx);
        int non_zero_count = 0;

        // TODO: profile the performance and try vectorizing with BLAS (or at::)
        for (int i_idx = 0; i_idx < inner_num; ++i_idx) {
            int i = idx[i_idx];
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
                int j = idx[j_idx];
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

// C++ interface
std::vector<at::Tensor> nmsfilt_forward(
    at::Tensor bbs, at::Tensor conf,
    float nms_threshold, int classes, float pre_threshold, int first_class, int max_output_boxes) {
  CHECK_INPUT_CPU(bbs);
  CHECK_INPUT_CPU(conf);

  AT_ASSERTM(bbs.dim() >= 2, "invalid bbs dim");
  AT_ASSERTM(conf.dim() >= 2, "invalid conf dim");
  int bbs_axis = bbs.dim() - 1;
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

  auto top_conf = conf.clone();

  if (pre_threshold >= 0) {
    AT_DISPATCH_FLOATING_TYPES(conf.scalar_type(), "nmsfilt_forward::pre_filter", ([&] {
      pre_filter(outer_num, channels, inner_num, classes, first_class,
                 pre_threshold,
                 top_conf.data<scalar_t>());
    }));
  }

  if (nms_threshold <= 0 || inner_num == 1)
      return {top_conf};

  AT_DISPATCH_FLOATING_TYPES(conf.scalar_type(), "nmsfilt_forward::nms_filter", ([&] {
    nms_filter(bbs.data<scalar_t>(),
               outer_num, channels, inner_num, classes, first_class, max_output_boxes,
               nms_threshold,
               top_conf.data<scalar_t>());
  }));

  return {top_conf};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &nmsfilt_forward, "NMSFilter forward (CPU)");
}

at::Tensor nmsfilt(
    at::Tensor bbs, at::Tensor conf,
    const double nms_threshold, const double pre_threshold, const int64_t max_output_boxes) {
    return nmsfilt_forward(bbs, conf, nms_threshold, 1, pre_threshold, 0, max_output_boxes)[0];
}

static auto registry = torch::RegisterOperators()
  .op("mtorch_ops::nmsfilt(Tensor bbs, Tensor conf,"
      "float nms_threshold, float pre_threshold, int max_output_boxes) -> Tensor",
      &nmsfilt);
