// Exact per-edge induced K1,3 (3-star) counts.
//
// For an oriented edge (center u, leaf v), S = N(u) \\ N[v] (v itself excluded).
// Each 4-node induced K1,3 centered at u using edge (u,v) is exactly a non-edge
// {a,b} inside S.  Therefore
//
//     k_{u,v} = C(|S|, 2) - e(S).
//
// The per-edge count is k_{u,v} + k_{v,u}; every star containing the edge has
// exactly one of the two endpoints as its center.  Counts are exact; no sampling.
//
// Build:
//   g++ -O3 -march=native -fopenmp analysis_sid_recall/k13_exact.cpp \
//       -o results/research/tiger_cluster_graph/k13_exact
// Run:
//   results/research/tiger_cluster_graph/k13_exact
// Self-test:
//   results/research/tiger_cluster_graph/k13_exact --selftest
//
// Input is the same undirected edge CSV used for PGD.  Output is an edge CSV
// with the original pair order and the exact K1,3 participation count.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

const char *kInput =
    "results/research/tiger_cluster_graph/pgd_behavior_edges.csv";
const char *kOutput =
    "results/research/tiger_cluster_graph/k13_edges.csv";
const char *kSummary =
    "results/research/tiger_cluster_graph/k13_summary.txt";

struct Graph {
    int n = 0;
    int words = 0;
    std::vector<std::pair<int, int>> edges;
    std::vector<std::vector<int>> adj;
    std::vector<std::vector<uint64_t>> bits;

    bool adjacent(int left, int right) const {
        return (bits[left][right >> 6] >> (right & 63)) & 1ULL;
    }

    void build_bits() {
        words = (n + 63) / 64;
        bits.assign(n, std::vector<uint64_t>(words, 0));
        for (const auto &edge : edges) {
            bits[edge.first][edge.second >> 6] |= 1ULL << (edge.second & 63);
            bits[edge.second][edge.first >> 6] |= 1ULL << (edge.first & 63);
        }
    }
};

Graph read_graph(const std::string &path) {
    Graph graph;
    std::ifstream input(path);
    if (!input) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(2);
    }
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        int left = -1, right = -1;
        std::sscanf(line.c_str(), "%d %d", &left, &right);
        if (left < 0 || right < 0 || left == right) {
            continue;
        }
        graph.edges.emplace_back(left, right);
        graph.n = std::max(graph.n, std::max(left, right) + 1);
    }
    graph.adj.assign(graph.n, {});
    for (const auto &edge : graph.edges) {
        graph.adj[edge.first].push_back(edge.second);
        graph.adj[edge.second].push_back(edge.first);
    }
    graph.build_bits();
    return graph;
}

long long k13_for_edge(const Graph &graph, int edge_index, int center, int leaf) {
    const auto &neighbours = graph.adj[center];
    std::vector<int> selected;
    selected.reserve(neighbours.size());
    for (int other : neighbours) {
        if (other == leaf || graph.adjacent(leaf, other)) {
            continue;
        }
        selected.push_back(other);
    }
    const long long size = static_cast<long long>(selected.size());
    if (size < 2) {
        return 0;
    }

    // Sparse bitset of S, with the non-empty word list reused for popcount.
    std::vector<uint64_t> selected_bits(graph.words, 0);
    std::vector<int> used_words;
    used_words.reserve(static_cast<size_t>(size));
    for (int vertex : selected) {
        const int word = vertex >> 6;
        if (selected_bits[word] == 0) {
            used_words.push_back(word);
        }
        selected_bits[word] |= 1ULL << (vertex & 63);
    }

    long long non_edges = 0;
    const long long pairs = size * (size - 1) / 2;
    const long long bit_cost = size * static_cast<long long>(used_words.size());
    if (pairs <= bit_cost) {
        for (size_t first = 0; first < selected.size(); ++first) {
            for (size_t second = first + 1; second < selected.size(); ++second) {
                if (!graph.adjacent(selected[first], selected[second])) {
                    ++non_edges;
                }
            }
        }
    } else {
        long long directed_edges = 0;
        for (int vertex : selected) {
            const auto &row = graph.bits[vertex];
            for (int word : used_words) {
                directed_edges += __builtin_popcountll(row[word] & selected_bits[word]);
            }
        }
        // The selected set contributes no self-loops; every edge is counted twice.
        const long long edges = directed_edges / 2;
        non_edges = pairs - edges;
    }
    (void)edge_index;
    return non_edges;
}

std::vector<long long> exact_counts(const Graph &graph) {
    std::vector<long long> counts(graph.edges.size(), 0);
    const long long edge_count = static_cast<long long>(graph.edges.size());
#pragma omp parallel
    {
        std::vector<long long> local(static_cast<size_t>(edge_count), 0);
#pragma omp for schedule(dynamic, 64)
        for (long long index = 0; index < edge_count; ++index) {
            const auto &edge = graph.edges[static_cast<size_t>(index)];
            const int first = edge.first;
            const int second = edge.second;
            const long long first_as_center = k13_for_edge(graph, static_cast<int>(index), first, second);
            const long long second_as_center = k13_for_edge(graph, static_cast<int>(index), second, first);
            local[static_cast<size_t>(index)] = first_as_center + second_as_center;
        }
#pragma omp critical
        {
            for (size_t index = 0; index < counts.size(); ++index) {
                counts[index] += local[index];
            }
        }
    }
    return counts;
}

long long brute_force_edge_count(const Graph &graph, int edge_index) {
    const int n = graph.n;
    long long total = 0;
    std::vector<int> vertices;
    for (int a = 0; a < n; ++a) {
        for (int b = a + 1; b < n; ++b) {
            for (int c = b + 1; c < n; ++c) {
                for (int d = c + 1; d < n; ++d) {
                    const int subset[4] = {a, b, c, d};
                    int degrees[4] = {0, 0, 0, 0};
                    bool edges[4][4] = {};
                    for (int i = 0; i < 4; ++i) {
                        for (int j = i + 1; j < 4; ++j) {
                            edges[i][j] = edges[j][i] = graph.adjacent(subset[i], subset[j]);
                            if (edges[i][j]) {
                                ++degrees[i];
                                ++degrees[j];
                            }
                        }
                    }
                    int ones = 0, threes = 0, center = -1;
                    for (int i = 0; i < 4; ++i) {
                        if (degrees[i] == 1) ++ones;
                        if (degrees[i] == 3) { ++threes; center = i; }
                    }
                    if (ones != 3 || threes != 1) {
                        continue;
                    }
                    bool contains_edge = false;
                    const auto &query = graph.edges[static_cast<size_t>(edge_index)];
                    for (int i = 0; i < 4; ++i) {
                        if (subset[i] == query.first) {
                            for (int j = 0; j < 4; ++j) {
                                if (subset[j] == query.second && i != j) contains_edge = true;
                            }
                        }
                    }
                    if (contains_edge) {
                        ++total;
                    }
                }
            }
        }
    }
    return total;
}

Graph make_graph(int n, const std::vector<std::pair<int, int>> &edges) {
    Graph graph;
    graph.n = n;
    graph.edges = edges;
    graph.adj.assign(n, {});
    for (const auto &edge : edges) {
        graph.adj[edge.first].push_back(edge.second);
        graph.adj[edge.second].push_back(edge.first);
    }
    graph.build_bits();
    return graph;
}

bool self_test() {
    struct Case {
        std::string name;
        int n;
        std::vector<std::pair<int, int>> edges;
        std::vector<long long> expected;
    };
    std::vector<Case> cases;
    cases.push_back({"K1,3", 4, {{0, 1}, {0, 2}, {0, 3}}, {1, 1, 1}});
    cases.push_back({"K1,4", 5, {{0, 1}, {0, 2}, {0, 3}, {0, 4}}, {3, 3, 3, 3}});
    cases.push_back({"K4", 4, {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}, {0, 0, 0, 0, 0, 0}});
    cases.push_back({"triangle+pendant", 4, {{0, 1}, {1, 2}, {2, 0}, {2, 3}}, {0, 0, 0, 0}});
    cases.push_back({"C4", 4, {{0, 1}, {1, 2}, {2, 3}, {3, 0}}, {0, 0, 0, 0}});
    for (const auto &test : cases) {
        Graph graph = make_graph(test.n, test.edges);
        const auto counts = exact_counts(graph);
        if (counts != test.expected) {
            std::cerr << "synthetic failure: " << test.name << "\n";
            return false;
        }
    }

    std::mt19937 rng(12345);
    for (int trial = 0; trial < 30; ++trial) {
        const int n = 9;
        std::vector<std::pair<int, int>> edges;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (std::uniform_real_distribution<double>(0, 1)(rng) < 0.28) {
                    edges.emplace_back(i, j);
                }
            }
        }
        Graph graph = make_graph(n, edges);
        const auto counts = exact_counts(graph);
        for (size_t index = 0; index < edges.size(); ++index) {
            const long long expected = brute_force_edge_count(graph, static_cast<int>(index));
            if (counts[index] != expected) {
                std::cerr << "random failure trial " << trial << " edge " << index
                          << " got " << counts[index] << " expected " << expected << "\n";
                return false;
            }
        }
    }
    std::cout << "K1,3 self-test: all synthetic and random cases pass\n";
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "--selftest") {
        return self_test() ? 0 : 1;
    }
    const std::string input_path = argc > 1 ? argv[1] : kInput;
    const std::string output_path = argc > 2 ? argv[2] : kOutput;
    const std::string summary_path = argc > 3 ? argv[3] : kSummary;
    const Graph graph = read_graph(input_path);
    const auto counts = exact_counts(graph);
    long long total = 0;
    for (long long value : counts) {
        total += value;
    }
    if (total % 3 != 0) {
        std::cerr << "K1,3 total is not divisible by 3: " << total << "\n";
        return 1;
    }
    std::ofstream output(output_path);
    output << "src,dst,k13\n";
    for (size_t index = 0; index < graph.edges.size(); ++index) {
        output << graph.edges[index].first << "," << graph.edges[index].second
               << "," << counts[index] << "\n";
    }
    std::ofstream summary(summary_path);
    summary << "nodes," << graph.n << "\n";
    summary << "edges," << graph.edges.size() << "\n";
    summary << "sum_edge_counts," << total << "\n";
    summary << "stars," << total / 3 << "\n";
    summary << "max_edge_count," << *std::max_element(counts.begin(), counts.end()) << "\n";
    std::cout << "wrote " << output_path << " and " << summary_path << "\n";
    return 0;
}
