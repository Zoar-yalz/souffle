/**
 * Simple test for TransitiveRelation
 * 
 * Reads edges from a CSV file and outputs the transitive closure.
 * 
 * Usage: ./test_transitive <input.csv> [output.csv]
 * 
 * CSV format: one edge per line, "from,to" (integers)
 * Example:
 *   1,2
 *   2,3
 *   3,4
 */

// Force use of local headers before system headers
#include "src/include/souffle/utility/ContainerUtil.h"
#include "src/include/souffle/datastructure/Graph.h"
#include "src/include/souffle/datastructure/TransitiveRelation.h"
#include "src/include/souffle/RamTypes.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// Define a Tuple type that matches what TransitiveRelation expects
template <typename T, std::size_t N>
struct TestTuple : public std::array<T, N> {
    // TransitiveRelation uses TupleType::value_type for the vertex type
    using value_type = T;
    static constexpr std::size_t arity = N;
    
    TestTuple() : std::array<T, N>{} {}
    TestTuple(std::initializer_list<T> init) {
        std::size_t i = 0;
        for (auto v : init) {
            if (i < N) (*this)[i++] = v;
        }
    }
};

using Tuple2 = TestTuple<souffle::RamDomain, 2>;
using TR = souffle::TransitiveRelation<Tuple2>;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.csv> [output.csv]\n";
        std::cerr << "CSV format: from,to (one edge per line)\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cerr << "Error: cannot open file " << argv[1] << "\n";
        return 1;
    }

    // Optional output file
    std::ofstream outfile;
    bool hasOutput = (argc >= 3);
    if (hasOutput) {
        outfile.open(argv[2]);
        if (!outfile) {
            std::cerr << "Error: cannot open output file " << argv[2] << "\n";
            return 1;
        }
    }

    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::milliseconds;

    TR rel;
    std::string line;
    int edgeCount = 0;

    std::cerr << "=== Reading edges from " << argv[1] << " ===\n";

    auto t0 = Clock::now();

    while (std::getline(infile, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string fromStr, toStr;

        // Support both comma and tab as delimiter
        char delim = (line.find(',') != std::string::npos) ? ',' : '\t';
        
        if (!std::getline(iss, fromStr, delim) || !std::getline(iss, toStr, delim)) {
            std::cerr << "Warning: skipping malformed line: " << line << "\n";
            continue;
        }

        try {
            souffle::RamDomain from = std::stol(fromStr);
            souffle::RamDomain to = std::stol(toStr);
            rel.insert(from, to);
            edgeCount++;
        } catch (const std::exception& e) {
            std::cerr << "Warning: skipping invalid line: " << line << "\n";
        }
    }

    auto t1 = Clock::now();
    std::cerr << "Inserted " << edgeCount << " edges in "
              << std::chrono::duration_cast<Ms>(t1 - t0).count() << " ms\n";

    // Trigger SCC computation
    auto t2 = Clock::now();
    rel.computeScc();
    auto t3 = Clock::now();
    std::cerr << "Number of SCCs: " << rel.getNumberOfSccs() << "\n";
    std::cerr << "SCC computation: " << std::chrono::duration_cast<Ms>(t3 - t2).count() << " ms\n";

    // Compute reachability (triggers reachability cache)
    auto t4 = Clock::now();
    rel.computeSccDagReachability();
    auto t5 = Clock::now();
    std::cerr << "SCC DAG reachability: " << std::chrono::duration_cast<Ms>(t5 - t4).count() << " ms\n";

    // Output transitive closure
    std::cerr << "Enumerating transitive closure...\n";
    auto t6 = Clock::now();
    std::size_t closureSize = 0;
    
    std::ostream& out = hasOutput ? outfile : std::cout;
    
    for (const auto& tuple : rel) {
        out << tuple[0] << "," << tuple[1] << "\n";
        closureSize++;
    }
    auto t7 = Clock::now();
    
    std::cerr << "Total pairs in closure: " << closureSize << "\n";
    std::cerr << "Enumeration + output: " << std::chrono::duration_cast<Ms>(t7 - t6).count() << " ms\n";
    std::cerr << "Total time: " << std::chrono::duration_cast<Ms>(t7 - t0).count() << " ms\n";
    
    if (hasOutput) {
        std::cerr << "Output written to " << argv[2] << "\n";
        outfile.close();
    }

    return 0;
}
