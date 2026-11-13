// How fast can this GPU walk stage 1's enumeration at all?
//
// The CPU spends 84.75 s serially in ORCA's stage 1, which for every node x walks the
// triples of its neighbours and tests three adjacencies per triple.  That loop is the
// floor any port must beat, so this program measures only the walk: it builds the same
// bit-packed adjacency on the device and counts, per node, how many triples have at
// least two internal edges.  No table is written -- the count is what the table would
// hold -- so the number printed is the best case for the enumeration part alone.
//
//   nvcc -O3 -arch=sm_89 -o build_common_bench build_common_bench.cu
//   ./build_common_bench graph.in
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

static __host__ __device__ inline bool adjacent_bit(const uint32_t *mat, int n, int x, int y) {
    size_t bit = (size_t)x * n + y;
    return mat[bit >> 5] & (1u << (bit & 31));
}

// One block per node, neighbours staged in shared memory.  A hub of degree d costs
// C(d,3) triple visits inside a single block, so blocks are split by the index of the
// outermost neighbour: each block takes a contiguous slice of the a-range.
__global__ void count_triples(const int *csr_ptr, const int *csr_adj, const uint32_t *mat,
                              int n, int slices, unsigned long long *visits,
                              unsigned long long *kept) {
    int x = blockIdx.x / slices;
    int slice = blockIdx.x % slices;
    int begin = csr_ptr[x], deg = csr_ptr[x + 1] - begin;
    if (slice >= deg) return;

    unsigned long long local_visits = 0, local_kept = 0;
    // the a-index range this block owns
    int a_from = (int)((long long)deg * slice / (slices < deg ? slices : deg));
    int a_to = (int)((long long)deg * (slice + 1) / (slices < deg ? slices : deg));

    for (int ia = a_from + threadIdx.x; ia < a_to; ia += blockDim.x) {
        int a = csr_adj[begin + ia];
        for (int ib = ia + 1; ib < deg; ib++) {
            int b = csr_adj[begin + ib];
            bool ab = adjacent_bit(mat, n, a, b);
            for (int ic = ib + 1; ic < deg; ic++) {
                int c = csr_adj[begin + ic];
                int st = (int)ab + (int)adjacent_bit(mat, n, a, c) + (int)adjacent_bit(mat, n, b, c);
                local_visits++;
                local_kept += (st >= 2);
            }
        }
    }
    atomicAdd(&visits[x], local_visits);
    atomicAdd(&kept[x], local_kept);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s graph.in\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    int n, m;
    if (fscanf(f, "%d %d", &n, &m) != 2) { fprintf(stderr, "bad header\n"); return 1; }
    std::vector<int> ea(m), eb(m), deg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        if (fscanf(f, "%d %d", &ea[i], &eb[i]) != 2) { fprintf(stderr, "bad edge %d\n", i); return 1; }
        deg[ea[i]]++; deg[eb[i]]++;
    }
    fclose(f);

    std::vector<int> ptr(n + 1, 0);
    for (int i = 0; i < n; i++) ptr[i + 1] = ptr[i] + deg[i];
    std::vector<int> fill(ptr.begin(), ptr.end() - 1), adj(ptr[n]);
    for (int i = 0; i < m; i++) {
        adj[fill[ea[i]]++] = eb[i];
        adj[fill[eb[i]]++] = ea[i];
    }
    for (int i = 0; i < n; i++) std::sort(adj.begin() + ptr[i], adj.begin() + ptr[i + 1]);

    size_t words = ((size_t)n * n + 31) / 32;
    std::vector<uint32_t> mat(words, 0);
    for (int x = 0; x < n; x++)
        for (int i = ptr[x]; i < ptr[x + 1]; i++) {
            size_t bit = (size_t)x * n + adj[i];
            mat[bit >> 5] |= 1u << (bit & 31);
        }
    fprintf(stderr, "graph %d nodes %d edges, bit matrix %.1f MB\n", n, m, words * 4.0 / 1048576.0);

    int *d_ptr, *d_adj; uint32_t *d_mat;
    unsigned long long *d_visits, *d_kept;
    CHECK(cudaMalloc(&d_ptr, (n + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_adj, m * 2 * sizeof(int)));
    CHECK(cudaMalloc(&d_mat, words * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_visits, n * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_kept, n * sizeof(unsigned long long)));
    CHECK(cudaMemcpy(d_ptr, ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_adj, adj.data(), m * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mat, mat.data(), words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_visits, 0, n * sizeof(unsigned long long)));
    CHECK(cudaMemset(d_kept, 0, n * sizeof(unsigned long long)));

    int slices = 8;
    dim3 grid((long)n * slices), block(128);
    auto t0 = std::chrono::steady_clock::now();
    count_triples<<<grid, block>>>(d_ptr, d_adj, d_mat, n, slices, d_visits, d_kept);
    CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    std::vector<unsigned long long> visits(n), kept(n);
    CHECK(cudaMemcpy(visits.data(), d_visits, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(kept.data(), d_kept, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    unsigned long long total_visits = 0, total_kept = 0;
    for (int i = 0; i < n; i++) { total_visits += visits[i]; total_kept += kept[i]; }
    double secs = std::chrono::duration<double>(t1 - t0).count();
    printf("triple visits %llu (%.3e), with >=2 internal edges %llu (%.3e)\n",
           total_visits, (double)total_visits, total_kept, (double)total_kept);
    printf("GPU walk: %.3f s  (%.2f G visits/s)\n", secs, total_visits / secs / 1e9);
    return 0;
}
