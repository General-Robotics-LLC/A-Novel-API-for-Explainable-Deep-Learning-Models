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



#include <ATen/ops/cholesky_ops.h>

namespace at {


// aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & cholesky_out(at::Tensor & out, const at::Tensor & self, bool upper=false) {
    return at::_ops::cholesky_out::call(self, upper, out);
}
// aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & cholesky_outf(const at::Tensor & self, bool upper, at::Tensor & out) {
    return at::_ops::cholesky_out::call(self, upper, out);
}

// aten::cholesky(Tensor self, bool upper=False) -> Tensor
inline at::Tensor cholesky(const at::Tensor & self, bool upper=false) {
    return at::_ops::cholesky::call(self, upper);
}

}
