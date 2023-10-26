#pragma once
// @generated by torchgen/gen.py from DispatchKeyFunction.h

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace compositeexplicitautograd {

TORCH_API ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _embedding_bag_out(at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq=false, int64_t mode=0, bool sparse=false, const c10::optional<at::Tensor> & per_sample_weights={}, bool include_last_offset=false, int64_t padding_idx=-1);
TORCH_API ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _embedding_bag_outf(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3);

} // namespace compositeexplicitautograd
} // namespace at
