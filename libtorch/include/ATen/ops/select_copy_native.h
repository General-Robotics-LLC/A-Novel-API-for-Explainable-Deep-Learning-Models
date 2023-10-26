#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API at::Tensor & select_copy_int_out_symint(const at::Tensor & self, int64_t dim, c10::SymInt index, at::Tensor & out);
TORCH_API at::Tensor select_copy_sparse_csr(const at::Tensor & self, int64_t dim, int64_t index);
TORCH_API at::Tensor select_copy_symint(const at::Tensor & self, int64_t dim, c10::SymInt index);
} // namespace native
} // namespace at
