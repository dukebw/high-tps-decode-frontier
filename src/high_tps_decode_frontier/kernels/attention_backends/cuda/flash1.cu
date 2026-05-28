#include "flash1_cuda.cuh"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cfloat>
#include <cmath>

constexpr int WARP_SIZE{32};
constexpr int Q_TILE{16};
constexpr int KV_TILE{64};
constexpr int HEAD_DIM{64};

constexpr __host__ __device__ int ceildiv(int x, int y) {
  return (x + y - 1) / y;
}

namespace tile {

template <int ROWS, int HEAD_DIM, int numThreads>
static __device__ void loadRows(__nv_bfloat16 *dst, const __nv_bfloat16 *src,
                                int baseRow, int totalRows) {
  for (int elem = threadIdx.x; elem < ROWS * HEAD_DIM; elem += numThreads) {
    int localRow = elem / HEAD_DIM;
    int head = elem % HEAD_DIM;
    int globalRow = baseRow + localRow;

    if (globalRow < totalRows) {
      dst[localRow * HEAD_DIM + head] = src[globalRow * HEAD_DIM + head];
    } else {
      dst[localRow * HEAD_DIM + head] = __float2bfloat16(0.0f);
    }
  }
}

template <int numElems, int numThreads>
static __device__ void fillRows(float *dst, float value) {
  for (int elem = threadIdx.x; elem < numElems; elem += numThreads)
    dst[elem] = value;
}

template <int Q_TILE, int KV_TILE, int HEAD_DIM, int numThreads>
static __device__ void mma(float *scoreTile, const __nv_bfloat16 *qTile,
                           const __nv_bfloat16 *kTile) {
  // ...
}

template <int row, int col> __device__ void max(float *x, const float *y) {
  // updates x in place
  // x: [row]
  // y: [row, col]
  // ...
}

template <int row, int col> __device__ void sum(float *out, const float *x) {
  // x: [row, col]
  // out: [row]
  // ...
}

} // namespace tile

template <int Q_TILE, int KV_TILE, int HEAD_DIM, int elemsPerThread,
          int numThreads>
__global__ void
flash1AttentionKernel(__nv_bfloat16 *out, const __nv_bfloat16 *q,
                      const __nv_bfloat16 *k, const __nv_bfloat16 *v,
                      int qSeqLen, int kvSeqLen) {
  // q: [qSeqLen, headDim]
  // k: [kvSeqLen, headDim]
  // v: [kvSeqLen, headDim]
  // out: thread block per [Q_TILE, headDim] output tile

  // TODO: this maps a single output element per thread, which isn't great
  // because it reduces the amount of work per thread (maybe?).
  // Lots of different (Q_TILE, HEAD_DIM, numThreads) scenarios.
  // Which are realistic?
  // Assume: numThreads > Q_TILE (plausible) and numThreads > HEAD_DIM (also
  // plausible).
  // Total elements: Q_TILE*HEAD_DIM.
  // Each thread is responsible for ceildiv(Q_TILE*HEAD_DIM, numThreads)
  // elements.
  int qBaseRow = blockIdx.x * Q_TILE;

  // online softmax state:
  // acc: [Q_TILE, HEAD_DIM]
  //   running unnormalized output
  // l: [Q_TILE] (running denominator)
  // m: [Q_TILE] (running max)
  __shared__ __nv_bfloat16 qTile[Q_TILE][HEAD_DIM];
  __shared__ __nv_bfloat16 kTile[KV_TILE][HEAD_DIM];
  __shared__ __nv_bfloat16 vTile[KV_TILE][HEAD_DIM];
  __shared__ float scoreTile[Q_TILE][KV_TILE];
  __shared__ float alpha[Q_TILE];

  __shared__ float acc[Q_TILE][HEAD_DIM];
  tile::fillRows<Q_TILE * HEAD_DIM, numThreads>(acc, /*value=*/0.0f);

  __shared__ float l[Q_TILE];
  tile::fillRows<Q_TILE, numThreads>(l, /*value=*/0.0f);

  __shared__ float m[Q_TILE];
  tile::fillRows<Q_TILE, numThreads>(m, /*value=*/-FLT_MAX);

  // for K_i, V_i: [KV_TILE, HEAD_DIM] blocks in zip(k, v)
  //   S_i := Q_i @ K_i.T * scale: [Q_TILE, KV_TILE] (score tile)
  //   m = max(m_old, rowmax(S_i)): [Q_TILE]
  //   P_i := exp(S_i - m): [Q_TILE, KV_TILE] (unnormalized probs tile)
  //   O_i := P_i @ V_i: [Q_TILE, headDim]

  //   update online softmax state:
  //   alpha := exp(m_old - m): [Q_TILE]
  //   l = alpha*l_old + rowsum(P_i): [Q_TILE]
  //   acc = alpha * acc + O_i: [Q_TILE, headDim]
  // out := acc/l

  tile::loadRows<Q_TILE, HEAD_DIM, numThreads>(qTile, q, qBaseRow, qSeqLen);
  // no barrier since there'll be a barrier before use

  int numKvBlocks = ceildiv(kvSeqLen, KV_TILE);
  for (int kvBlockIdx = 0; kvBlockIdx < numKvBlocks; ++kvBlockIdx) {
    int kvBaseRow = kvBlockIdx * KV_TILE;
    tile::loadRows<KV_TILE, HEAD_DIM, numThreads>(kTile, k, kvBaseRow,
                                                  kvSeqLen);
    __syncthreads();

    tile::loadRows<KV_TILE, HEAD_DIM, numThreads>(vTile, v, kvBaseRow,
                                                  kvSeqLen);
    __syncthreads();

    // S_i := Q_i @ K_i.T * scale
    tile::mma<Q_TILE, KV_TILE, HEAD_DIM, numThreads>(scoreTile, qTile, kTile);

    // m = max(m_old, rowmax(S_i))
    tile::max<Q_TILE, KV_TILE>(m, scoreTile);

    // in place update
    int sKvIdx = threadIdx.x % KV_TILE;
    int sQIdx = threadIdx.x / KV_TILE;
    if (sQIdx < Q_TILE && sKvIdx < KV_TILE)
      scoreTile[sQIdx][sKvIdx] = expf(scoreTile[sQIdx][sKvIdx] - m[sQIdx]);
    __syncthreads();

    // O_i := P_i @ V_i
    tile::mma<Q_TILE, KV_TILE, HEAD_DIM>(acc, scoreTile, vTile);

    // alpha := exp(m_old - m)

    // l = alpha*l_old + rowsum(P_i)

    // acc = alpha * acc + O_i
  }

  // out := acc/l
}

extern "C" cudaError_t flash1AttentionCudaLaunch(void *out, const void *q,
                                                 const void *k, const void *v,
                                                 int qSeqLen, int kvSeqLen,
                                                 int headDim,
                                                 cudaStream_t stream) {
  constexpr int elemsPerThread{4};
  constexpr int numThreads{Q_TILE * HEAD_DIM / elemsPerThread};
  int numBlocks{ceildiv(qSeqLen, Q_TILE)};

  return flash1AttentionKernel<Q_TILE, KV_TILE, HEAD_DIM, elemsPerThread,
                               numThreads>
      <<<numBlocks, numThreads, /*dynSmemBytes=*/0, stream>>>(
          out, q, k, v, qSeqLen, kvSeqLen);
}
