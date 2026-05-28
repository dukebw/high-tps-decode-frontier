#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

extern "C" cudaError_t flash1AttentionCudaLaunch(void *out, const void *q,
                                                  const void *k, const void *v,
                                                  int qSeqLen, int kvSeqLen,
                                                  int headDim,
                                                  cudaStream_t stream);
