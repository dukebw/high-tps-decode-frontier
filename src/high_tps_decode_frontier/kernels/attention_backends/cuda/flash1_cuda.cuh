#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

extern "C" cudaError_t flash1_attention_cuda_launch(void *out, const void *q,
                                                    const void *k, const void *v,
                                                    std::size_t numel,
                                                    std::size_t element_size,
                                                    cudaStream_t stream);
