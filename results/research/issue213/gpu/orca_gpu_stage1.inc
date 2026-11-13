// Stage 1 of ORCA's count5, on the GPU.
//
// The CPU version walks, for every node, all triples of its neighbours and tests three
// adjacencies per triple: 8.66e9 visits, 84.75 s single-threaded, and it is the serial
// part that no amount of OpenMP threads can shrink.  Measured on the same graph, the
// GPU walks all of it in 0.883 s (bench in build_common_bench.cu) and only 58.1M of the
// visits pass the >=2-internal-edge test, so the tables are small:
//
//   common2[(a,b)]   = number of common neighbours of a and b        (n*n uint32, 2.4 GB)
//   common3[(a,b,c)] = number of common neighbours of the triple     (open addressing)
//
// Both are built by the same enumeration, so the counts are the same counts, and the
// full run is validated by comparing its output with the single-threaded one byte for
// byte.  Lookups from stage 2/3 stay on the CPU, against these flat tables instead of
// unordered_map -- which is also why they get cheaper.

static int *g_ptr = nullptr, *g_adjlist = nullptr;
static int *g_ptr_host = nullptr;   // CSR offsets on the host, for stage 3's edge ids
static uint32_t *g_mat = nullptr;
static unsigned *g_c2_dev = nullptr;
static unsigned long long *g_keys = nullptr;
static unsigned *g_vals = nullptr;
static size_t g_cap = 0;

static unsigned *h_c2 = nullptr;            // n*n
static unsigned long long *h_keys = nullptr;
static unsigned *h_vals = nullptr;
static long long h_entries = 0;

__device__ inline bool mat_bit(const uint32_t *mat, int n, int x, int y) {
    size_t bit = (size_t)x * n + y;
    return mat[bit >> 5] & (1u << (bit & 31));
}

__host__ __device__ inline unsigned long long triple_key(int a, int b, int c) {
    return ((unsigned long long)a << 42) | ((unsigned long long)b << 21) | (unsigned long long)c;
}

// The slot came from (key * K) & mask at first, and that only ever looks at the low bits of the
// key: with key = a<<42 | b<<21 | c, every triple sharing (b mod 64, c) landed in one slot, probe
// chains ran past the bound and 1.49M triples were silently dropped.  A 64-bit finaliser spreads
// the whole key before the mask.
__host__ __device__ static inline size_t scramble(unsigned long long x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (size_t)x;
}

__device__ inline void hash_inc(unsigned long long *keys, unsigned *vals, size_t mask,
                                unsigned long long key) {
    size_t h = scramble(key) & mask;
    for (int probes = 0; probes < 4096; probes++) {
        unsigned long long cur = keys[h];
        if (cur == key) { atomicAdd(&vals[h], 1u); return; }
        if (cur == 0ull) {
            unsigned long long prev = atomicCAS(&keys[h], 0ull, key);
            // A lost CAS can mean the winner put the SAME key here; adding to the next free slot
            // instead would leave the key in two places and every lookup would see only one of
            // them, which is exactly the small undercount a first version produced.
            if (prev == 0ull || prev == key) { atomicAdd(&vals[h], 1u); return; }
            continue;                              // a different key won the slot: keep probing
        }
        h = (h + 1) & mask;
    }
}

__global__ void k_build_tables(const int *ptr, const int *adj, const uint32_t *mat, int n,
                               int slices, unsigned *c2, unsigned long long *keys, unsigned *vals,
                               size_t mask, unsigned long long *overflow) {
    int x = blockIdx.x / slices;
    int slice = blockIdx.x % slices;
    int begin = ptr[x], deg = ptr[x + 1] - begin;
    int used = slices < deg ? slices : deg;
    if (slice >= used) return;
    int a_from = (int)((long long)deg * slice / used);
    int a_to = (int)((long long)deg * (slice + 1) / used);

    for (int ia = a_from + threadIdx.x; ia < a_to; ia += blockDim.x) {
        int a = adj[begin + ia];
        for (int ib = ia + 1; ib < deg; ib++) {
            int b = adj[begin + ib];
            atomicAdd(&c2[(size_t)a * n + b], 1u);
            bool ab = mat_bit(mat, n, a, b);
            for (int ic = ib + 1; ic < deg; ic++) {
                int c = adj[begin + ic];
                int st = (int)ab + (int)mat_bit(mat, n, a, c) + (int)mat_bit(mat, n, b, c);
                if (st >= 2) hash_inc(keys, vals, mask, triple_key(a, b, c));
            }
        }
    }
    (void)overflow;
}

__host__ static void gpu_tables_init(int n, int m) {
    // CSR from ORCA's edge list
    std::vector<int> deg(n, 0);
    for (int i = 0; i < m; i++) { deg[edges[i].a]++; deg[edges[i].b]++; }
    std::vector<int> ptr(n + 1, 0), fill(n, 0);
    for (int i = 0; i < n; i++) ptr[i + 1] = ptr[i] + deg[i];
    for (int i = 0; i < n; i++) fill[i] = ptr[i];
    std::vector<int> adjlist((size_t)m * 2);
    for (int i = 0; i < m; i++) {
        adjlist[fill[edges[i].a]++] = edges[i].b;
        adjlist[fill[edges[i].b]++] = edges[i].a;
    }
    for (int i = 0; i < n; i++) std::sort(adjlist.begin() + ptr[i], adjlist.begin() + ptr[i + 1]);

    size_t words = ((size_t)n * n + 31) / 32;
    g_cap = ((size_t)1) << 27;                     // 134M slots: 58.1M insert events at most
    h_c2 = (unsigned*)calloc((size_t)n * n, sizeof(unsigned));
    h_keys = (unsigned long long*)calloc(g_cap, sizeof(unsigned long long));
    h_vals = (unsigned*)calloc(g_cap, sizeof(unsigned));

    g_ptr_host = (int*)malloc((n + 1) * sizeof(int));
    memcpy(g_ptr_host, ptr.data(), (n + 1) * sizeof(int));
    cudaMalloc(&g_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&g_adjlist, (size_t)m * 2 * sizeof(int));
    cudaMalloc(&g_mat, words * sizeof(uint32_t));
    cudaMalloc(&g_c2_dev, (size_t)n * n * sizeof(unsigned));
    cudaMalloc(&g_keys, g_cap * sizeof(unsigned long long));
    cudaMalloc(&g_vals, g_cap * sizeof(unsigned));
    cudaMemcpy(g_ptr, ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_adjlist, adjlist.data(), (size_t)m * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat, adj_matrix, words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(g_c2_dev, 0, (size_t)n * n * sizeof(unsigned));
    cudaMemset(g_keys, 0, g_cap * sizeof(unsigned long long));
    cudaMemset(g_vals, 0, g_cap * sizeof(unsigned));
}

__host__ static void gpu_build_tables(int n, int m) {
    int slices = 8;
    dim3 grid((long)n * slices), block(128);
    k_build_tables<<<grid, block>>>(g_ptr, g_adjlist, g_mat, n, slices, g_c2_dev, g_keys, g_vals,
                                    g_cap - 1, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c2, g_c2_dev, (size_t)n * n * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_keys, g_keys, g_cap * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals, g_vals, g_cap * sizeof(unsigned), cudaMemcpyDeviceToHost);
    long long entries = 0;
    for (size_t i = 0; i < g_cap; i++) entries += (h_keys[i] != 0ull);
    h_entries = entries;
    fprintf(stderr, "gpu tables: common3 entries %lld, capacity %zu\n", entries, g_cap);
    if (getenv("GENE_DUMP")) {
        FILE *f2 = fopen("gpu_common2.bin", "wb");
        unsigned long long cnt2 = 0;
        for (size_t i = 0; i < (size_t)n * n; i++) cnt2 += (h_c2[i] != 0u);
        fwrite(&cnt2, 8, 1, f2);
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++) {
                unsigned v = h_c2[(size_t)a * n + b];
                if (v) { int key[2] = {a, b}; fwrite(key, 4, 2, f2); fwrite(&v, 4, 1, f2); }
            }
        fclose(f2);
        FILE *f3 = fopen("gpu_common3.bin", "wb");
        fwrite(&entries, 8, 1, f3);
        for (size_t h = 0; h < g_cap; h++) {
            if (!h_keys[h]) continue;
            unsigned long long k = h_keys[h];
            int key[3] = {(int)(k >> 42), (int)((k >> 21) & ((1u << 21) - 1)), (int)(k & ((1u << 21) - 1))};
            fwrite(key, 4, 3, f3); fwrite(&h_vals[h], 4, 1, f3);
        }
        fclose(f3);
        fprintf(stderr, "dumped gpu tables\n");
    }
}

__host__ static inline unsigned common2_lookup(int a, int b) {
    if (a > b) { int t = a; a = b; b = t; }
    return h_c2[(size_t)a * n + b];
}

__host__ static inline unsigned common3_lookup(int a, int b, int c) {
    // TRIPLE stores the sorted triple
    unsigned long long key = triple_key(a, b, c);
    size_t mask = g_cap - 1;
    size_t h = scramble(key) & mask;
    for (; h_keys[h] != 0ull; h = (h + 1) & mask)
        if (h_keys[h] == key) return h_vals[h];
    return 0;
}
