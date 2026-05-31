#include "flash1_cuda.cuh"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cfloat>
#include <cmath>

constexpr int WARP_SIZE = 32;
constexpr int Q_TILE = 16;
constexpr int KV_TILE = 64;
constexpr int HEAD_DIM = 64;

constexpr __host__ __device__ int ceildiv(int x, int y) {
  return (x + y - 1) / y;
}

namespace tile {

template <int ROWS, int HEAD_DIM, int numThreads, bool isTransposed = false>
static __device__ void loadRows(__nv_bfloat16 *dst, const __nv_bfloat16 *src,
                                int baseRow, int totalRows) {
  for (int elem = threadIdx.x; elem < ROWS * HEAD_DIM; elem += numThreads) {
    int localRow = elem / HEAD_DIM;
    int head = elem % HEAD_DIM;
    int globalRow = baseRow + localRow;

    int dstIdx;
    if constexpr (isTransposed)
      dstIdx = head * ROWS + localRow;
    else
      dstIdx = localRow * HEAD_DIM + head;

    if (globalRow < totalRows) {
      dst[dstIdx] = src[globalRow * HEAD_DIM + head];
    } else {
      dst[dstIdx] = __float2bfloat16(0.0f);
    }
  }
}

template <int numElems, int numThreads>
static __device__ void fill(float *dst, float value) {
  for (int elem = threadIdx.x; elem < numElems; elem += numThreads)
    dst[elem] = value;
}

static __device__ float toFloat(float value) { return value; }

static __device__ float toFloat(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <int M, int K, int N, int numThreads, typename AType, typename BType>
static __device__ void mma(float *c, const AType *a, const BType *b) {
  for (int elem = threadIdx.x; elem < M * N; elem += numThreads) {
    int mIdx = elem / N;
    int nIdx = elem % N;

    float accum = 0.0f;
    for (int kIdx = 0; kIdx < K; ++kIdx)
      accum += toFloat(a[mIdx * K + kIdx]) * toFloat(b[kIdx * N + nIdx]);

    c[mIdx * N + nIdx] = accum;
  }
}

template <int row, int col, int numThreads>
__device__ void max(float *x, const float *y) {
  // updates x in place
  // x: [row]
  // y: [row, col]
  // x and y already in smem
  // block reduce max
  static_assert(numThreads % WARP_SIZE == 0);

  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int numWarps = numThreads / WARP_SIZE;
  constexpr int rowsPerWarp = ceildiv(row, numWarps);
  int rowBase = warp * rowsPerWarp;

  // reduce in registers first
  for (int rowIdx = rowBase; rowIdx < (rowBase + rowsPerWarp); ++rowIdx) {
    if (rowIdx >= row)
      continue;

    float localMax = -FLT_MAX;
    for (int colIdx = lane; colIdx < col; colIdx += WARP_SIZE)
      localMax = fmaxf(localMax, y[(rowIdx * col) + colIdx]);

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
      localMax = fmaxf(localMax, __shfl_xor_sync(0xFFFFFFFF, localMax, offset));

    if (lane == 0)
      x[rowIdx] = fmaxf(x[rowIdx], localMax);
  }
}

template <int row, int col, int numThreads>
__device__ void sum(float *out, const float *x) {
  // x: [row, col]
  // out: [row]
  static_assert(numThreads % WARP_SIZE == 0);

  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int numWarps = numThreads / WARP_SIZE;
  constexpr int rowsPerWarp = ceildiv(row, numWarps);
  int rowBaseIdx = warp * rowsPerWarp;

  for (int rowIdx = rowBaseIdx; rowIdx < rowBaseIdx + rowsPerWarp; ++rowIdx) {
    if (rowIdx >= row)
      continue;

    float localSum = 0.0f;
    for (int colIdx = lane; colIdx < col; colIdx += WARP_SIZE)
      localSum += x[rowIdx * col + colIdx];

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
      localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, offset);

    if (lane == 0)
      out[rowIdx] = localSum;
  }
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
  __shared__ __nv_bfloat16 kTile[HEAD_DIM][KV_TILE];
  __shared__ __nv_bfloat16 vTile[KV_TILE][HEAD_DIM];
  __shared__ float oTile[Q_TILE][HEAD_DIM];
  __shared__ float scoreTile[Q_TILE][KV_TILE];
  __shared__ float rowSumP_i[Q_TILE];
  __shared__ float alpha[Q_TILE];

  __shared__ float acc[Q_TILE][HEAD_DIM];
  tile::fill<Q_TILE * HEAD_DIM, numThreads>(acc, /*value=*/0.0f);

  __shared__ float l[Q_TILE];
  tile::fill<Q_TILE, numThreads>(l, /*value=*/0.0f);

  __shared__ float m[Q_TILE];
  tile::fill<Q_TILE, numThreads>(m, /*value=*/-FLT_MAX);
  __syncthreads();

  // for K_i, V_i: [KV_TILE, HEAD_DIM] blocks in zip(k, v)
  //   S_i := Q_i @ K_i.T * scale: [Q_TILE, KV_TILE] (score tile)
  //   m = max(m_prev, rowmax(S_i)): [Q_TILE]
  //   P_i := exp(S_i - m): [Q_TILE, KV_TILE] (unnormalized probs tile)
  //   O_i := P_i @ V_i: [Q_TILE, headDim]

  //   update online softmax state:
  //   alpha := exp(m_prev - m): [Q_TILE]
  //   l = alpha*l_old + rowsum(P_i): [Q_TILE]
  //   acc = alpha * acc + O_i: [Q_TILE, headDim]
  // out := acc/l

  tile::loadRows<Q_TILE, HEAD_DIM, numThreads>(qTile, q, qBaseRow, qSeqLen);
  // no barrier since there'll be a barrier before use

  int numKvBlocks = ceildiv(kvSeqLen, KV_TILE);
  for (int kvBlockIdx = 0; kvBlockIdx < numKvBlocks; ++kvBlockIdx) {
    int kvBaseRow = kvBlockIdx * KV_TILE;
    tile::loadRows<KV_TILE, HEAD_DIM, numThreads, /*isTransposed=*/true>(
        kTile, k, kvBaseRow, kvSeqLen);
    tile::loadRows<KV_TILE, HEAD_DIM, numThreads>(vTile, v, kvBaseRow,
                                                  kvSeqLen);
    __syncthreads();

    // S_i := Q_i @ K_i.T * scale
    tile::mma<Q_TILE, HEAD_DIM, KV_TILE, numThreads>(scoreTile, qTile, kTile);

    // m = max(m_prev, rowmax(S_i))
    float mPrev = -FLT_MAX;
    if (threadIdx.x < Q_TILE)
      mPrev = m[threadIdx.x];
    tile::max<Q_TILE, KV_TILE, numThreads>(m, scoreTile);
    __syncthreads();

    // in place update
    int sKvIdx = threadIdx.x % KV_TILE;
    int sQIdx = threadIdx.x / KV_TILE;
    if (sQIdx < Q_TILE && sKvIdx < KV_TILE)
      scoreTile[sQIdx][sKvIdx] = expf(scoreTile[sQIdx][sKvIdx] - m[sQIdx]);
    __syncthreads();

    // O_i := P_i @ V_i
    tile::mma<Q_TILE, KV_TILE, HEAD_DIM, numThreads>(oTile, scoreTile, vTile);

    // alpha := exp(m_prev - m)
    tile::sum<Q_TILE, KV_TILE, numThreads>(rowSumP_i, scoreTile);
    if (threadIdx.x < Q_TILE) {
      alpha[threadIdx.x] = expf(mPrev - m[threadIdx.x]);

      // l = alpha*l_old + rowsum(P_i)
      l[threadIdx.x] =
          alpha[threadIdx.x] * l[threadIdx.x] + rowSumP_i[threadIdx.x];
    }
    __syncthreads();

    // acc = alpha * acc + O_i
    for (int elem = threadIdx.x; elem < Q_TILE * HEAD_DIM; elem += numThreads) {
      int localRow = elem / HEAD_DIM;
      int head = elem % HEAD_DIM;
      int globalRow = qBaseRow + localRow;

      if (globalRow < qSeqLen)
        acc[localRow][head] =
            alpha[localRow] * acc[localRow][head] + oTile[localRow][head];
    }
  }

  // out := acc/l
  for (int elem = threadIdx.x; elem < Q_TILE * HEAD_DIM; elem += numThreads) {
    int localRow = elem / HEAD_DIM;
    int head = elem % HEAD_DIM;
    int globalRow = qBaseRow + localRow;

    if (globalRow < qSeqLen)
      out[globalRow * HEAD_DIM + head] = acc[localRow][head] / l[localRow];
  }
}

extern "C" cudaError_t flash1AttentionCudaLaunch(void *out, const void *q,
                                                 const void *k, const void *v,
                                                 int qSeqLen, int kvSeqLen,
                                                 int headDim,
                                                 cudaStream_t stream) {
  constexpr int elemsPerThread = 4;
  constexpr int numThreads = Q_TILE * HEAD_DIM / elemsPerThread;
  int numBlocks = ceildiv(qSeqLen, Q_TILE);

  return flash1AttentionKernel<Q_TILE, KV_TILE, HEAD_DIM, elemsPerThread,
                               numThreads>
      <<<numBlocks, numThreads, /*dynSmemBytes=*/0, stream>>>(
          out, q, k, v, qSeqLen, kvSeqLen);
}
