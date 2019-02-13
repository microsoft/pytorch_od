// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef REGION_COMMON_HPP_
#define REGION_COMMON_HPP_

template <typename scalar_t>
struct TBox {
    scalar_t x, y, w, h;
};

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__ __forceinline__
#else
#define CUDA_HOSTDEV
#endif

template <typename scalar_t>
CUDA_HOSTDEV scalar_t TOverlap(scalar_t x1, scalar_t w1, scalar_t x2, scalar_t w2)
{
    auto l1 = x1 - w1 / 2;
    auto l2 = x2 - w2 / 2;
    auto left = l1 > l2 ? l1 : l2;
    auto r1 = x1 + w1 / 2;
    auto r2 = x2 + w2 / 2;
    auto right = r1 < r2 ? r1 : r2;
    return right - left;
}

template <typename scalar_t>
CUDA_HOSTDEV scalar_t TBoxIntersection(scalar_t ax, scalar_t ay, scalar_t aw, scalar_t ah,
        scalar_t bx, scalar_t by, scalar_t bw, scalar_t bh) {
    auto w = TOverlap(ax, aw, bx, bw);
    auto h = TOverlap(ay, ah, by, bh);
    if (w < 0 || h < 0) {
        return 0;
    }
    else {
        return w * h;
    }
}

template <typename scalar_t>
CUDA_HOSTDEV scalar_t TBoxIou(scalar_t ax, scalar_t ay, scalar_t aw, scalar_t ah,
        scalar_t bx, scalar_t by, scalar_t bw, scalar_t bh) {
    auto i = TBoxIntersection(ax, ay, aw, ah, bx, by, bw, bh);
    auto u = aw * ah + bw * bh - i;
    return i / u;
}

#endif  // REGION_COMMON_HPP_
