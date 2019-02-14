// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef MTORCH_COMMON_HPP_
#define MTORCH_COMMON_HPP_
/**
* Common checks for all the extensions
**/

// Work-around ATen regression
#ifndef AT_ASSERTM
#define AT_ASSERTM AT_ASSERT
#endif

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

#endif  // MTORCH_COMMON_HPP_
