// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/torch.h>

#include <vector>
#include <stack>
#include <float.h>

#include "mtorch_common.h"

namespace {

struct Pred {
    double parent_p;
    int parent_argmax;
    int g;
};

template <typename scalar_t>
void predict_tree_stack(int outer_num, int channels, int inner_num,
                        bool append_max,
                        float threshold,
                        const int* group_offset_data, const int* group_size_data, const int* child_data, const int* child_size_data,
                        const scalar_t* obj_data, const scalar_t* prob_data,
                        int max_stack_size, int n, int s, int g,
                        scalar_t* top_data, bool output_tree_path) {
    std::stack<Pred> preds;
    scalar_t obj = obj_data ? obj_data[n * inner_num + s] : 1;
    double root_p = output_tree_path ? obj : 1.0;
    // if it is output_tree_path, the score should be the obj * category_prob
    // in the path
    threshold = output_tree_path ? (threshold * obj) : threshold;
    preds.push({ root_p, -1, g });
    const int top_channels = append_max ? (channels + 1) : channels;
    while (!preds.empty()) {
        assert(preds.size() <= max_stack_size);
        auto pred = preds.top();
        preds.pop();
        double p = pred.parent_p;
        int argmax = 0;
        {
            g = pred.g;
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
                for (int sg = 0; sg < sg_count; ++sg)
                    preds.push({ p, argmax, g + sg });
                continue;
            }
        } else {
            argmax = pred.parent_argmax;
            if (argmax < 0)
                continue;
            p = pred.parent_p;
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
                // used as the indicator for class-independent NMS. otherwise, the
                // maximum value will always be the ones in the first
                // child-level of the root node.
                top_data[max_idx] = obj;
            } else {
                if (node_p > top_data[max_idx]) {
                    top_data[max_idx] = node_p;
                }
            }
        }
    }
}

} // namespace

// C++ interface
std::vector<at::Tensor> smtpred_forward(
    at::Tensor conf, at::Tensor obj,
    at::Tensor group_offset, at::Tensor group_size, at::Tensor child, at::Tensor child_size,
    float threshold, bool output_tree_path, bool append_max,
    int root_size, int stack_size
) {

  CHECK_INPUT_CPU(conf);
  CHECK_INPUT_CPU(obj);
  CHECK_INPUT_CPU(group_offset);
  CHECK_INPUT_CPU(group_size);
  CHECK_INPUT_CPU(child);
  CHECK_INPUT_CPU(child_size);

  AT_ASSERTM(conf.dim() >= 2, "invalid conf dim");

  int outer_num = conf.size(0);
  int inner_num = 1;
  for (int i = 2; i < conf.dim(); ++i)
    inner_num *= conf.size(i);

  auto shape = conf.sizes().vec();
  int channels = shape[1];
  if (append_max)
    shape[1] = channels + 1;

  auto top = at::zeros(shape, conf.type());

  root_size++;

  AT_DISPATCH_FLOATING_TYPES(conf.type(), "smtpred_forward", ([&] {
    auto group_offset_data = group_offset.data<int>();
    auto group_size_data = group_size.data<int>();
    auto child_data = child.data<int>();
    auto child_size_data = child_size.data<int>();
    scalar_t* obj_data = nullptr;
    if (obj.numel())
      obj_data = obj.data<scalar_t>();
    auto prob_data = conf.data<scalar_t>();
    auto top_data = top.data<scalar_t>();
    for (int index = 0; index < outer_num * root_size * inner_num; ++index) {
        const int s = index % inner_num;
        const int g = (index / inner_num) % root_size;
        const int n = (index / inner_num) / root_size;

        predict_tree_stack(outer_num, channels, inner_num,
                           append_max,
                           threshold,
                           group_offset_data, group_size_data, child_data, child_size_data,
                           obj_data, prob_data,
                           stack_size, n, s, g,
                           top_data,
                           output_tree_path);
    }
  }));

  return {top};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &smtpred_forward, "SoftmaxTreePrediction forward (CPU)");
}
