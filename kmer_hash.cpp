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

static size_t S;                            // Total hash table size
static size_t local_size;                   // Local hash table size per process
static std::vector<std::vector<kmer_pair>> local_table; // Local hash table with chaining
static std::vector<kmer_pair> start_kmers;  // Local start k-mers

int main(int argc, char** argv) {
    upcxx::init();

    if (argc < 2) {
        BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test [prefix]]\n");
        upcxx::finalize();
        exit(1);
    }

    std::string kmer_fname = std::string(argv[1]);
    std::string run_type = "";

    if (argc >= 3) {
        run_type = std::string(argv[2]);
    }

    std::string test_prefix = "test";
    if (run_type == "test" && argc >= 4) {
        test_prefix = std::string(argv[3]);
    }

    int ks = kmer_size(kmer_fname);

    if (ks != KMER_LEN) {
        throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
                                 "-mers, while this binary is compiled for " +
                                 std::to_string(KMER_LEN) +
                                 "-mers.  Modify packing.hpp and recompile.");
    }

    size_t n_kmers = line_count(kmer_fname);
    size_t P = upcxx::rank_n();
    local_size = (n_kmers * 2 + P - 1) / P; // Ceiling division for load factor 0.5
    S = local_size * P;
    local_table.resize(local_size);

    if (run_type == "verbose") {
        BUtil::print("Initializing hash table of size %d for %d kmers.\n", S,
                     n_kmers);
    }

    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, P, upcxx::rank_me());
    if (run_type == "verbose") {
        BUtil::print("Finished reading kmers.\n");
    }

    upcxx::barrier();

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

    auto start_read = std::chrono::high_resolution_clock::now();

    // Initialize contigs and active indices
    std::vector<std::list<kmer_pair>> contigs;
    for (const auto& start_kmer : start_kmers) {
        contigs.push_back({start_kmer});
    }
    std::list<size_t> active_indices;
    for (size_t i = 0; i < contigs.size(); i++) {
        active_indices.push_back(i);
    }

    // Extend contigs with batched fetches
    while (!active_indices.empty()) {
        // Group k-mers to fetch by owner process
        std::vector<std::vector<pkmer_t>> to_send(P);
        std::vector<std::vector<size_t>> contig_indices(P);
        for (auto it = active_indices.begin(); it != active_indices.end(); ) {
            size_t idx = *it;
            const auto& last_kmer = contigs[idx].back();
            if (last_kmer.forwardExt() != 'F') {
                pkmer_t next_kmer = last_kmer.next_kmer();
                size_t h = next_kmer.hash();
                size_t slot = h % S;
                size_t owner = slot / local_size;
                to_send[owner].push_back(next_kmer);
                contig_indices[owner].push_back(idx);
                ++it;
            } else {
                it = active_indices.erase(it);
            }
        }

        // Send batched RPCs
        std::vector<upcxx::future<std::vector<kmer_pair>>> futures;
        std::vector<int> owners;
        for (int owner = 0; owner < P; owner++) {
            if (!to_send[owner].empty()) {
                futures.push_back(upcxx::rpc(owner, [](std::vector<pkmer_t> kmers) {
                    std::vector<kmer_pair> results;
                    for (const auto& kmer : kmers) {
                        size_t h = kmer.hash();
                        size_t slot = h % S;
                        size_t local_slot = slot % local_size;
                        for (const auto& stored : local_table[local_slot]) {
                            if (stored.kmer == kmer) {
                                results.push_back(stored);
                                break;
                            }
                        }
                    }
                    if (results.size() != kmers.size()) {
                        throw std::runtime_error("Some k-mers not found in batch fetch");
                    }
                    return results;
                }, to_send[owner]));
                owners.push_back(owner);
            }
        }

        // Process fetched k-mers
        for (size_t i = 0; i < futures.size(); i++) {
            int owner = owners[i];
            auto fetched = futures[i].wait();
            for (size_t j = 0; j < fetched.size(); j++) {
                contigs[contig_indices[owner][j]].push_back(fetched[j]);
            }
        }
    }

    auto end_read = std::chrono::high_resolution_clock::now();
    upcxx::barrier();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> read = end_read - start_read;
    std::chrono::duration<double> insert = end_insert - start;
    std::chrono::duration<double> total = end - start;

    int numKmers = std::accumulate(
        contigs.begin(), contigs.end(), 0,
        [](int sum, const std::list<kmer_pair>& contig) { return sum + contig.size(); });

    if (run_type != "test") {
        BUtil::print("Assembled in %lf total\n", total.count());
    }
    if (run_type == "verbose") {
        printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
               " (%lf read, %lf insert, %lf total)\n",
               upcxx::rank_me(), contigs.size(), numKmers, start_kmers.size(), read.count(),
               insert.count(), total.count());
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