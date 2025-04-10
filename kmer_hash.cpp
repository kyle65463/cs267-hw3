#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <upcxx/upcxx.hpp>
#include <vector>

#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "butil.hpp"

// Static variables for the distributed hash table
static size_t S;                            // Total hash table size
static size_t local_size;                   // Local hash table size per process
static std::vector<std::list<kmer_pair>> local_table; // Local hash table with chaining
static std::vector<kmer_pair> start_kmers;  // Local start k-mers

/** Find a k-mer in the distributed hash table.
 * @param key_kmer The k-mer to find.
 * @return The kmer_pair containing the k-mer and its extensions.
 */
kmer_pair find_kmer(const pkmer_t& key_kmer) {
    size_t h = key_kmer.hash();
    size_t slot = h % S;
    size_t owner = slot / local_size;
    size_t local_slot = slot % local_size;
    auto future = upcxx::rpc(owner, [](pkmer_t key_kmer, size_t local_slot) {
        for (const auto& kmer : local_table[local_slot]) {
            if (kmer.kmer == key_kmer) {
                return kmer;
            }
        }
        throw std::runtime_error("K-mer not found");
    }, key_kmer, local_slot);
    return future.wait();
}

int main(int argc, char** argv) {
    upcxx::init();

    // Parse command-line arguments
    if (argc < 2) {
        BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test [prefix]]\n");
        upcxx::finalize();
        exit(1);
    }

    std::string kmer_fname = argv[1];
    std::string run_type = (argc >= 3) ? argv[2] : "";
    std::string test_prefix = (run_type == "test" && argc >= 4) ? argv[3] : "test";

    // Validate k-mer size
    int ks = kmer_size(kmer_fname);
    if (ks != KMER_LEN) {
        throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
                                 "-mers, while this binary is compiled for " +
                                 std::to_string(KMER_LEN) +
                                 "-mers. Modify packing.hpp and recompile.");
    }

    // Compute hash table sizes
    size_t n_kmers = line_count(kmer_fname);
    size_t P = upcxx::rank_n();
    local_size = (n_kmers * 2 + P - 1) / P; // Ceiling division for load factor 0.5
    S = local_size * P;                      // Ensure S is divisible by P
    local_table.resize(local_size);

    if (run_type == "verbose") {
        BUtil::print("Initializing distributed hash table with total size %zu, local size %zu for %zu kmers.\n",
                     S, local_size, n_kmers);
    }

    // Read local portion of k-mers
    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, P, upcxx::rank_me());
    if (run_type == "verbose") {
        BUtil::print("Finished reading kmers.\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Batch insert k-mers
    std::vector<std::vector<kmer_pair>> kmers_to_send(P);
    for (const auto& kmer : kmers) {
        size_t h = kmer.hash();
        size_t slot = h % S;
        size_t owner = slot / local_size;
        kmers_to_send[owner].push_back(kmer);
    }

    std::vector<upcxx::future<>> futures;
    for (int owner = 0; owner < P; owner++) {
        if (!kmers_to_send[owner].empty()) {
            futures.push_back(upcxx::rpc(owner, [](std::vector<kmer_pair> kmers) {
                for (const auto& kmer : kmers) {
                    size_t h = kmer.hash();
                    size_t slot = h % S;
                    size_t local_slot = slot % local_size;
                    local_table[local_slot].push_back(kmer);
                    if (kmer.backwardExt() == 'F') {
                        start_kmers.push_back(kmer);
                    }
                }
            }, kmers_to_send[owner]));
        }
    }
    upcxx::when_all(futures).wait();

    auto end_insert = std::chrono::high_resolution_clock::now();
    upcxx::barrier();
    double insert_time = std::chrono::duration<double>(end_insert - start).count();
    if (run_type != "test") {
        BUtil::print("Finished inserting in %lf\n", insert_time);
    }
    upcxx::barrier();

    // Build contigs from local start k-mers
    auto start_read = std::chrono::high_resolution_clock::now();
    std::list<std::list<kmer_pair>> contigs;
    for (const auto& start_kmer : start_kmers) {
        std::list<kmer_pair> contig;
        contig.push_back(start_kmer);
        while (contig.back().forwardExt() != 'F') {
            pkmer_t next_kmer = contig.back().next_kmer();
            kmer_pair val_kmer = find_kmer(next_kmer);
            contig.push_back(val_kmer);
        }
        contigs.push_back(contig);
    }

    auto end_read = std::chrono::high_resolution_clock::now();
    upcxx::barrier();
    auto end = std::chrono::high_resolution_clock::now();

    // Compute timing and statistics
    std::chrono::duration<double> read = end_read - start_read;
    std::chrono::duration<double> insert = end_insert - start;
    std::chrono::duration<double> total = end - start;
    int numKmers = std::accumulate(contigs.begin(), contigs.end(), 0,
        [](int sum, const std::list<kmer_pair>& contig) { return sum + contig.size(); });

    if (run_type != "test") {
        BUtil::print("Assembled in %lf total\n", total.count());
    }
    if (run_type == "verbose") {
        printf("Rank %d reconstructed %zu contigs with %d nodes from %zu start nodes."
               " (%lf read, %lf insert, %lf total)\n",
               upcxx::rank_me(), contigs.size(), numKmers, start_kmers.size(),
               read.count(), insert.count(), total.count());
    }
    if (run_type == "test") {
        std::ofstream fout(test_prefix + "_" + std::to_string(upcxx::rank_me()) + ".dat");
        for (const auto& contig : contigs) {
            fout << extract_contig(contig) << std::endl;
        }
        fout.close();
    }

    upcxx::finalize();
    return 0;
}