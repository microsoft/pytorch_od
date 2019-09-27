// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <torch/script.h>

#include <vector>

#include "mtorch_common.h"
#include "region_common.hpp"

namespace {

template <typename scalar_t>
at::Tensor nms_kernel(
                const at::Tensor& bbs,
                const at::Tensor& conf,
                float thresh,
                int max_output_boxes) {
  auto w_t = bbs.select(1, 2).contiguous();
  auto h_t = bbs.select(1, 3).contiguous();

  auto x_t = bbs.select(1, 0).contiguous();
  auto y_t = bbs.select(1, 1).contiguous();

  at::Tensor areas_t = w_t * h_t;
  auto outer_num = bbs.size(0);
  at::Tensor suppressed_t = at::zeros({outer_num}, bbs.options().dtype(at::kByte).device(at::kCPU));
  auto order_t = std::get<1>(conf.sort(0, /* descending=*/true));


  auto bbs_data = bbs.data<scalar_t>();
  auto order_data = order_t.data<int64_t>();
  auto areas_data = areas_t.data<scalar_t>();
  auto suppressed_data = suppressed_t.data<uint8_t>();
  auto x_data = x_t.data<scalar_t>();
  auto y_data = y_t.data<scalar_t>();
  auto w_data = w_t.data<scalar_t>();
  auto h_data = h_t.data<scalar_t>();

  int non_zero_count = 0;
  for (int64_t i = 0; i < outer_num; i++) {
    auto i_idx = order_data[i];
    if (suppressed_data[i_idx])
      continue;
    if (non_zero_count == max_output_boxes) {
      // suppress the rest
      suppressed_data[i_idx] = 1;
      continue;
    }
    non_zero_count++;

    auto ax = x_data[i_idx];
    auto ay = y_data[i_idx];
    auto aw = w_data[i_idx];
    auto ah = h_data[i_idx];
    auto area = areas_data[i_idx];

    for (int64_t j = i + 1; j < outer_num; j++) {
      auto j_idx = order_data[j];
      if (suppressed_data[j_idx])
        continue;

      auto bx = x_data[j_idx];
      auto by = y_data[j_idx];
      auto bw = w_data[j_idx];
      auto bh = h_data[j_idx];
      auto inter = TBoxIntersection(ax, ay, aw, ah, bx, by, bw, bh);
      auto iou = inter / (area + areas_data[j_idx] - inter);
      if (iou > thresh)
        suppressed_data[j_idx] = 1;
   }
  }

  return at::nonzero(suppressed_t == 0).squeeze(1);
}

} // namespace

// C++ interface
at::Tensor nms_cpu_forward(
    at::Tensor bbs, at::Tensor conf,
    float nms_threshold, int max_output_boxes) {
  CHECK_INPUT_CPU(bbs);
  CHECK_INPUT_CPU(conf);

  if (bbs.numel() == 0 || nms_threshold <= 0) {
    return at::empty({0}, bbs.options().dtype(at::kLong).device(at::kCPU));
  }

  AT_ASSERTM(conf.dim() == 1, "invalid conf dim");
  AT_ASSERTM(bbs.dim() == 2, "invalid bbs dim");
  AT_ASSERTM(bbs.size(-1) == 4, "bbs axis must have 4 corners");
  AT_ASSERTM(bbs.numel() == conf.numel() * 4, "conf and bbs mismatch element count");

  at::Tensor keep;
  AT_DISPATCH_FLOATING_TYPES(conf.scalar_type(), "nms::nms_cpu_forward", ([&] {
    keep = nms_kernel<scalar_t>(bbs, conf, nms_threshold, max_output_boxes);
  }));

  return keep;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &nms_cpu_forward, "NMS forward (CPU)");
}

at::Tensor nms(
    at::Tensor bbs, at::Tensor conf,
    const double nms_threshold, const int64_t max_output_boxes) {
    return nms_cpu_forward(bbs, conf, nms_threshold, max_output_boxes);
}

static auto registry = torch::jit::RegisterOperators()
  .op("mtorch_ops::nms(Tensor bbs, Tensor conf,"
      "float nms_threshold, int max_output_boxes) -> Tensor",
      &nms);
