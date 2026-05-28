#include "flash1_cuda.cuh"

#include "ATen/cuda/CUDAContext.h"
#include "c10/cuda/CUDAException.h"
#include "torch/extension.h"

#include <limits>

torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(k.is_cuda(), "k must be CUDA");
  TORCH_CHECK(v.is_cuda(), "v must be CUDA");
  TORCH_CHECK(q.dim() == 2, "q must be [seq, head_dim]");
  TORCH_CHECK(k.dim() == 2, "k must be [seq, head_dim]");
  TORCH_CHECK(v.dim() == 2, "v must be [seq, head_dim]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
  TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q/v dtype mismatch");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16,
              "flash1 currently supports only bfloat16");
  TORCH_CHECK(q.size(1) == k.size(1), "q/k head_dim mismatch");
  TORCH_CHECK(k.size(0) == v.size(0), "k/v seq_len mismatch");
  TORCH_CHECK(k.size(1) == v.size(1), "k/v head_dim mismatch");

  auto q_contiguous = q.contiguous();
  auto k_contiguous = k.contiguous();
  auto v_contiguous = v.contiguous();
  auto out = torch::empty_like(q_contiguous);
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr auto maxInt = std::numeric_limits<int>::max();
  TORCH_CHECK(q_contiguous.size(0) <= maxInt, "q seq_len exceeds int range");
  TORCH_CHECK(k_contiguous.size(0) <= maxInt, "kv seq_len exceeds int range");
  TORCH_CHECK(q_contiguous.size(1) <= maxInt, "head_dim exceeds int range");

  C10_CUDA_CHECK(flash1AttentionCudaLaunch(
      out.data_ptr(), q_contiguous.data_ptr(), k_contiguous.data_ptr(),
      v_contiguous.data_ptr(), static_cast<int>(q_contiguous.size(0)),
      static_cast<int>(k_contiguous.size(0)),
      static_cast<int>(q_contiguous.size(1)), stream));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention", &attention, "FlashAttention 1 practice attention");
}
