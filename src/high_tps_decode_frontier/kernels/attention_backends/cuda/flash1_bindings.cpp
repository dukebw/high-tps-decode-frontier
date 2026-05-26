#include "flash1_cuda.cuh"

#include "ATen/cuda/CUDAContext.h"
#include "c10/cuda/CUDAException.h"
#include "torch/extension.h"

torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(k.is_cuda(), "k must be CUDA");
  TORCH_CHECK(v.is_cuda(), "v must be CUDA");
  TORCH_CHECK(q.dim() == 2, "q must be [seq, head_dim]");
  TORCH_CHECK(k.dim() == 2, "k must be [seq, head_dim]");
  TORCH_CHECK(v.dim() == 2, "v must be [seq, head_dim]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
  TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q/v dtype mismatch");
  TORCH_CHECK(q.size(1) == k.size(1), "q/k head_dim mismatch");
  TORCH_CHECK(k.size(0) == v.size(0), "k/v seq_len mismatch");
  TORCH_CHECK(k.size(1) == v.size(1), "k/v head_dim mismatch");

  auto q_contiguous = q.contiguous();
  auto k_contiguous = k.contiguous();
  auto v_contiguous = v.contiguous();
  auto out = torch::empty_like(q_contiguous);
  auto stream = at::cuda::getCurrentCUDAStream();

  C10_CUDA_CHECK(flash1_attention_cuda_launch(
      out.data_ptr(), q_contiguous.data_ptr(), k_contiguous.data_ptr(),
      v_contiguous.data_ptr(), static_cast<std::size_t>(out.numel()),
      static_cast<std::size_t>(out.element_size()), stream));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention", &attention, "FlashAttention 1 practice attention");
}
