// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CAFFE_CUDA_HPP_
#define CAFFE_CUDA_HPP_
/**
* Caffe CUDA port to PyTorch
**/

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 1024 threads per block
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#endif  // CAFFE_CUDA_HPP_
