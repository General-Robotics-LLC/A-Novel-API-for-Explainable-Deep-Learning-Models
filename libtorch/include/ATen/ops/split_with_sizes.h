#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/split_with_sizes_ops.h>

namespace at {


// aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
inline ::std::vector<at::Tensor> split_with_sizes(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim=0) {
    return at::_ops::split_with_sizes::call(self, c10::fromIntArrayRefSlow(split_sizes), dim);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  ::std::vector<at::Tensor> split_with_sizes(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim=0) {
    return at::_ops::split_with_sizes::call(self, c10::fromIntArrayRefSlow(split_sizes), dim);
  }
}

// aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
inline ::std::vector<at::Tensor> split_with_sizes_symint(const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim=0) {
    return at::_ops::split_with_sizes::call(self, split_sizes, dim);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  ::std::vector<at::Tensor> split_with_sizes(const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim=0) {
    return at::_ops::split_with_sizes::call(self, split_sizes, dim);
  }
}

}
