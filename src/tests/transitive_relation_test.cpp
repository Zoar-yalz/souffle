/*
 * Souffle - A Datalog Compiler
 * Copyright (c) 2025, The Souffle Developers. All rights reserved
 * Licensed under the Universal Permissive License v 1.0 as shown at:
 * - https://opensource.org/licenses/UPL
 * - <souffle root>/licenses/SOUFFLE-UPL.txt
 */

/************************************************************************
 *
 * @file transitive_relation_test.cpp
 *
 * Test cases for TransitiveRelation data structure.
 *
 ***********************************************************************/

#include "tests/test.h"

#include "souffle/datastructure/TransitiveRelation.h"
#include "souffle/RamTypes.h"
#include <cstddef>
#include <iostream>
#include <set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace souffle {
namespace test {

// Test tuple type
template <std::size_t Arity>
struct TestTuple : public std::array<RamDomain, Arity> {
    static constexpr std::size_t arity = Arity;
    using value_type = RamDomain;
};

using Tuple2 = TestTuple<2>;
using TR = TransitiveRelation<Tuple2>;

/** Basic scoping and construction **/
TEST(TransitiveRelationTest, Scoping) {
    TR rel;
    EXPECT_TRUE(rel.empty());
    EXPECT_EQ(rel.size(), 0);
}

/** Single edge insertion **/
TEST(TransitiveRelationTest, SingleEdge) {
    TR rel;
    
    rel.insert(1, 2);
    EXPECT_FALSE(rel.empty());
    
    // Reflexive-transitive closure should contain:
    // (1,1), (1,2), (2,2)
    EXPECT_TRUE(rel.contains(1, 1));
    EXPECT_TRUE(rel.contains(1, 2));
    EXPECT_TRUE(rel.contains(2, 2));
    EXPECT_FALSE(rel.contains(2, 1));
    
    EXPECT_EQ(rel.size(), 3);
}

/** Chain: 1 -> 2 -> 3 **/
TEST(TransitiveRelationTest, Chain) {
    TR rel;
    
    rel.insert(1, 2);
    rel.insert(2, 3);
    
    // Should contain all reflexive pairs and transitive pairs
    EXPECT_TRUE(rel.contains(1, 1));
    EXPECT_TRUE(rel.contains(2, 2));
    EXPECT_TRUE(rel.contains(3, 3));
    EXPECT_TRUE(rel.contains(1, 2));
    EXPECT_TRUE(rel.contains(2, 3));
    EXPECT_TRUE(rel.contains(1, 3));  // Transitive
    
    // Reverse should not exist
    EXPECT_FALSE(rel.contains(3, 1));
    EXPECT_FALSE(rel.contains(3, 2));
    EXPECT_FALSE(rel.contains(2, 1));
    
    // Size: (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) = 6
    EXPECT_EQ(rel.size(), 6);
}

/** Self-loop **/
TEST(TransitiveRelationTest, SelfLoop) {
    TR rel;
    
    rel.insert(1, 1);
    
    EXPECT_TRUE(rel.contains(1, 1));
    EXPECT_EQ(rel.size(), 1);
    
    // Adding another vertex
    rel.insert(2, 2);
    EXPECT_TRUE(rel.contains(2, 2));
    EXPECT_FALSE(rel.contains(1, 2));
    EXPECT_EQ(rel.size(), 2);
}

/** Cycle: 1 -> 2 -> 3 -> 1 **/
TEST(TransitiveRelationTest, Cycle) {
    TR rel;
    
    rel.insert(1, 2);
    rel.insert(2, 3);
    rel.insert(3, 1);
    
    // All vertices in cycle should reach each other
    EXPECT_TRUE(rel.contains(1, 2));
    EXPECT_TRUE(rel.contains(1, 3));
    EXPECT_TRUE(rel.contains(2, 1));
    EXPECT_TRUE(rel.contains(2, 3));
    EXPECT_TRUE(rel.contains(3, 1));
    EXPECT_TRUE(rel.contains(3, 2));
    
    // Reflexive
    EXPECT_TRUE(rel.contains(1, 1));
    EXPECT_TRUE(rel.contains(2, 2));
    EXPECT_TRUE(rel.contains(3, 3));
    
    // All pairs: 3 * 3 = 9
    EXPECT_EQ(rel.size(), 9);
}

/** Diamond graph: 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4 **/
TEST(TransitiveRelationTest, Diamond) {
    TR rel;
    
    rel.insert(1, 2);
    rel.insert(1, 3);
    rel.insert(2, 4);
    rel.insert(3, 4);
    
    // 1 can reach all
    EXPECT_TRUE(rel.contains(1, 2));
    EXPECT_TRUE(rel.contains(1, 3));
    EXPECT_TRUE(rel.contains(1, 4));
    
    // 2 and 3 can reach 4
    EXPECT_TRUE(rel.contains(2, 4));
    EXPECT_TRUE(rel.contains(3, 4));
    
    // 4 cannot reach others (except itself)
    EXPECT_FALSE(rel.contains(4, 1));
    EXPECT_FALSE(rel.contains(4, 2));
    EXPECT_FALSE(rel.contains(4, 3));
    
    // 2 and 3 cannot reach each other
    EXPECT_FALSE(rel.contains(2, 3));
    EXPECT_FALSE(rel.contains(3, 2));
    
    // Size: 4 reflexive + (1,2), (1,3), (1,4), (2,4), (3,4) = 4 + 5 = 9
    EXPECT_EQ(rel.size(), 9);
}

/** Clear operation **/
TEST(TransitiveRelationTest, Clear) {
    TR rel;
    
    rel.insert(1, 2);
    rel.insert(2, 3);
    EXPECT_FALSE(rel.empty());
    EXPECT_EQ(rel.size(), 6);
    
    rel.clear();
    EXPECT_TRUE(rel.empty());
    EXPECT_EQ(rel.size(), 0);
    
    // Can insert again after clear
    rel.insert(5, 6);
    EXPECT_TRUE(rel.contains(5, 6));
    EXPECT_EQ(rel.size(), 3);
}

/** Double clear should not crash **/
TEST(TransitiveRelationTest, DoubleClear) {
    TR rel;
    
    rel.insert(1, 2);
    rel.insert(2, 3);
    rel.clear();
    rel.clear();  // Second clear should be safe
    EXPECT_TRUE(rel.empty());
    
    rel.insert(1, 2);
    EXPECT_EQ(rel.size(), 3);
}

/** Non-existent vertices **/
TEST(TransitiveRelationTest, NonExistent) {
    TR rel;
    
    rel.insert(1, 2);
    
    // Querying non-existent vertices
    EXPECT_FALSE(rel.contains(99, 100));
    EXPECT_FALSE(rel.contains(1, 99));
    EXPECT_FALSE(rel.contains(99, 1));
}

/** SCC computation **/
TEST(TransitiveRelationTest, SccComputation) {
    TR rel;
    
    // Create two separate SCCs connected by an edge
    // SCC1: 1 <-> 2 (cycle)
    // SCC2: 3 <-> 4 (cycle)
    // Edge: 2 -> 3
    rel.insert(1, 2);
    rel.insert(2, 1);
    rel.insert(3, 4);
    rel.insert(4, 3);
    rel.insert(2, 3);
    
    EXPECT_EQ(rel.getNumberOfSccs(), 2);
    
    // Vertices in same SCC should have same SCC id
    EXPECT_EQ(rel.getSccOf(1), rel.getSccOf(2));
    EXPECT_EQ(rel.getSccOf(3), rel.getSccOf(4));
    EXPECT_NE(rel.getSccOf(1), rel.getSccOf(3));
    
    // SCC1 can reach SCC2, but not vice versa
    EXPECT_TRUE(rel.contains(1, 3));
    EXPECT_TRUE(rel.contains(1, 4));
    EXPECT_FALSE(rel.contains(3, 1));
    EXPECT_FALSE(rel.contains(4, 2));
}

/** Iterator - all pairs **/
TEST(TransitiveRelationTest, IteratorAll) {
    TR rel;
    
    rel.insert(1, 2);
    rel.insert(2, 3);
    
    std::set<std::pair<RamDomain, RamDomain>> pairs;
    for (const auto& t : rel) {
        pairs.insert({t[0], t[1]});
    }
    
    // Should have 6 pairs
    EXPECT_EQ(pairs.size(), 6);
    EXPECT_TRUE(pairs.count({1, 1}));
    EXPECT_TRUE(pairs.count({1, 2}));
    EXPECT_TRUE(pairs.count({1, 3}));
    EXPECT_TRUE(pairs.count({2, 2}));
    EXPECT_TRUE(pairs.count({2, 3}));
    EXPECT_TRUE(pairs.count({3, 3}));
}

/** Iterator - from specific source **/
TEST(TransitiveRelationTest, IteratorFrom) {
    TR rel;
    
    rel.insert(1, 2);
    rel.insert(2, 3);
    rel.insert(4, 5);
    
    // Get pairs from source 1
    auto range = rel.template getBoundaries<1>(Tuple2{1, 0});
    
    std::set<std::pair<RamDomain, RamDomain>> pairs;
    for (auto it = range.begin(); it != range.end(); ++it) {
        pairs.insert({(*it)[0], (*it)[1]});
    }
    
    // From 1: (1,1), (1,2), (1,3)
    EXPECT_EQ(pairs.size(), 3);
    EXPECT_TRUE(pairs.count({1, 1}));
    EXPECT_TRUE(pairs.count({1, 2}));
    EXPECT_TRUE(pairs.count({1, 3}));
}

/** Tuple-based insert and contains **/
TEST(TransitiveRelationTest, TupleAPI) {
    TR rel;
    
    Tuple2 t1 = {1, 2};
    Tuple2 t2 = {2, 3};
    
    rel.insert(t1);
    rel.insert(t2);
    
    EXPECT_TRUE(rel.contains(Tuple2{1, 2}));
    EXPECT_TRUE(rel.contains(Tuple2{1, 3}));
    EXPECT_TRUE(rel.contains(Tuple2{1, 1}));
    EXPECT_FALSE(rel.contains(Tuple2{3, 1}));
}

/** Large graph - linear chain **/
TEST(TransitiveRelationTest, LargeChain) {
    TR rel;
    constexpr std::size_t N = 1000;
    
    for (std::size_t i = 0; i < N - 1; ++i) {
        rel.insert(static_cast<RamDomain>(i), static_cast<RamDomain>(i + 1));
    }
    
    // First vertex can reach all
    EXPECT_TRUE(rel.contains(0, static_cast<RamDomain>(N - 1)));
    
    // Last cannot reach first
    EXPECT_FALSE(rel.contains(static_cast<RamDomain>(N - 1), 0));
    
    // Size should be N + (N-1) + (N-2) + ... + 1 = N*(N+1)/2
    EXPECT_EQ(rel.size(), N * (N + 1) / 2);
}

/** Tree structure **/
TEST(TransitiveRelationTest, Tree) {
    TR rel;
    
    // Binary tree:
    //       1
    //      / \
    //     2   3
    //    / \
    //   4   5
    rel.insert(1, 2);
    rel.insert(1, 3);
    rel.insert(2, 4);
    rel.insert(2, 5);
    
    // Root reaches all
    EXPECT_TRUE(rel.contains(1, 2));
    EXPECT_TRUE(rel.contains(1, 3));
    EXPECT_TRUE(rel.contains(1, 4));
    EXPECT_TRUE(rel.contains(1, 5));
    
    // Leaves don't reach siblings
    EXPECT_FALSE(rel.contains(4, 5));
    EXPECT_FALSE(rel.contains(3, 4));
}

/** Multiple disconnected components **/
TEST(TransitiveRelationTest, DisconnectedComponents) {
    TR rel;
    
    // Component 1: 1 -> 2
    rel.insert(1, 2);
    
    // Component 2: 10 -> 11 -> 12
    rel.insert(10, 11);
    rel.insert(11, 12);
    
    // No cross-component reachability
    EXPECT_FALSE(rel.contains(1, 10));
    EXPECT_FALSE(rel.contains(10, 1));
    EXPECT_FALSE(rel.contains(2, 12));
    
    // Within component works
    EXPECT_TRUE(rel.contains(1, 2));
    EXPECT_TRUE(rel.contains(10, 12));
    
    // SCCs: each vertex is its own SCC (no cycles)
    EXPECT_EQ(rel.getNumberOfSccs(), 5);
}

/** Duplicate edge insertion **/
TEST(TransitiveRelationTest, DuplicateEdge) {
    TR rel;
    
    EXPECT_TRUE(rel.insert(1, 2));   // New edge
    EXPECT_FALSE(rel.insert(1, 2)); // Duplicate
    EXPECT_FALSE(rel.insert(1, 2)); // Duplicate again
    
    EXPECT_EQ(rel.size(), 3);  // Still (1,1), (1,2), (2,2)
}

/** Partition for parallel iteration **/
TEST(TransitiveRelationTest, Partition) {
    TR rel;
    
    rel.insert(1, 2);
    rel.insert(2, 3);
    rel.insert(4, 5);
    
    auto partitions = rel.partition(4);
    
    // Should have partitions (one per source vertex)
    EXPECT_TRUE(partitions.size() > 0);
    
    // All pairs should be covered
    std::set<std::pair<RamDomain, RamDomain>> allPairs;
    for (const auto& range : partitions) {
        for (auto it = range.begin(); it != range.end(); ++it) {
            allPairs.insert({(*it)[0], (*it)[1]});
        }
    }
    
    EXPECT_EQ(allPairs.size(), rel.size());
}

/** SCC DAG reachability **/
TEST(TransitiveRelationTest, SccDagReachability) {
    TR rel;
    
    // Chain of SCCs: SCC1 -> SCC2 -> SCC3
    // SCC1: {1, 2} cycle
    // SCC2: {3, 4} cycle
    // SCC3: {5, 6} cycle
    rel.insert(1, 2);
    rel.insert(2, 1);
    rel.insert(3, 4);
    rel.insert(4, 3);
    rel.insert(5, 6);
    rel.insert(6, 5);
    
    // Connect SCCs
    rel.insert(2, 3);
    rel.insert(4, 5);
    
    rel.computeSccDagReachability();
    
    std::size_t scc1 = rel.getSccOf(1);
    std::size_t scc2 = rel.getSccOf(3);
    std::size_t scc3 = rel.getSccOf(5);
    
    EXPECT_TRUE(rel.sccReaches(scc1, scc2));
    EXPECT_TRUE(rel.sccReaches(scc1, scc3));
    EXPECT_TRUE(rel.sccReaches(scc2, scc3));
    
    EXPECT_FALSE(rel.sccReaches(scc3, scc1));
    EXPECT_FALSE(rel.sccReaches(scc3, scc2));
    EXPECT_FALSE(rel.sccReaches(scc2, scc1));
}

/** Empty iterator **/
TEST(TransitiveRelationTest, EmptyIterator) {
    TR rel;
    
    std::size_t count = 0;
    for (const auto& t : rel) {
        (void)t;
        ++count;
    }
    
    EXPECT_EQ(count, 0);
}

/** Size consistency with iterator **/
TEST(TransitiveRelationTest, SizeConsistency) {
    TR rel;
    
    rel.insert(1, 2);
    rel.insert(2, 3);
    rel.insert(3, 4);
    rel.insert(1, 5);
    
    std::size_t iterCount = 0;
    for (const auto& t : rel) {
        (void)t;
        ++iterCount;
    }
    
    EXPECT_EQ(rel.size(), iterCount);
}

#ifdef _OPENMP
/** Parallel insertion **/
TEST(TransitiveRelationTest, ParallelInsert) {
    TR rel;
    constexpr std::size_t N = 1000;
    
    // Insert chain in parallel (may cause race conditions if not handled)
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(N - 1); ++i) {
        rel.insert(static_cast<RamDomain>(i), static_cast<RamDomain>(i + 1));
    }
    
    // Basic sanity check: some edges should exist
    // Note: Parallel insertion may lead to inconsistent state,
    // so we only check that it doesn't crash
    EXPECT_FALSE(rel.empty());
}

/** Parallel contains queries **/
TEST(TransitiveRelationTest, ParallelContains) {
    TR rel;
    constexpr std::size_t N = 100;
    
    // Build a chain
    for (std::size_t i = 0; i < N - 1; ++i) {
        rel.insert(static_cast<RamDomain>(i), static_cast<RamDomain>(i + 1));
    }
    
    // Force SCC computation
    rel.getNumberOfSccs();
    
    // Parallel contains queries should be safe (read-only after SCC computed)
    std::atomic<std::size_t> trueCount{0};
    
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(N); ++i) {
        for (int j = 0; j < static_cast<int>(N); ++j) {
            if (rel.contains(static_cast<RamDomain>(i), static_cast<RamDomain>(j))) {
                ++trueCount;
            }
        }
    }
    
    EXPECT_EQ(trueCount.load(), rel.size());
}
#endif  // ifdef _OPENMP

/** Verify non-reflexive transitive closure calculation **/
TEST(TransitiveRelationTest, NonReflexiveSize) {
    TR rel;
    
    // Chain without cycles
    rel.insert(1, 2);
    rel.insert(2, 3);
    rel.insert(3, 4);
    
    // Reflexive-transitive closure: 4 + 3 + 2 + 1 = 10
    std::size_t reflexiveSize = rel.size();
    EXPECT_EQ(reflexiveSize, 10);
    
    // Non-reflexive: subtract 4 (one for each vertex without real cycle)
    // Expected: 10 - 4 = 6 pairs
    std::size_t nonReflexiveCount = 0;
    for (const auto& t : rel) {
        if (t[0] != t[1]) {
            ++nonReflexiveCount;
        }
    }
    EXPECT_EQ(nonReflexiveCount, 6);
}

/** Verify closure with cycles keeps self-pairs **/
TEST(TransitiveRelationTest, CycleSelfPairs) {
    TR rel;
    
    // Cycle: 1 -> 2 -> 3 -> 1
    rel.insert(1, 2);
    rel.insert(2, 3);
    rel.insert(3, 1);
    
    // All 9 pairs should exist including (1,1), (2,2), (3,3)
    EXPECT_EQ(rel.size(), 9);
    
    // Count pairs where a == b
    std::size_t selfPairs = 0;
    for (const auto& t : rel) {
        if (t[0] == t[1]) {
            ++selfPairs;
        }
    }
    EXPECT_EQ(selfPairs, 3);
    
    // In a cycle, these self-pairs represent real paths (a -> ... -> a)
    // So non-reflexive closure should also have 9 pairs
}

/** Mixed: cycle + chain **/
TEST(TransitiveRelationTest, MixedCycleChain) {
    TR rel;
    
    // Cycle: 1 <-> 2
    rel.insert(1, 2);
    rel.insert(2, 1);
    
    // Chain: 2 -> 3 -> 4
    rel.insert(2, 3);
    rel.insert(3, 4);
    
    // SCC analysis
    EXPECT_EQ(rel.getNumberOfSccs(), 3);  // {1,2}, {3}, {4}
    EXPECT_EQ(rel.getSccOf(1), rel.getSccOf(2));
    
    // Reachability
    EXPECT_TRUE(rel.contains(1, 4));   // 1 -> 2 -> 3 -> 4
    EXPECT_TRUE(rel.contains(2, 4));   // 2 -> 3 -> 4
    EXPECT_FALSE(rel.contains(3, 1)); // 3 cannot reach 1
    EXPECT_FALSE(rel.contains(4, 1)); // 4 cannot reach 1
    
    // Size:
    // From SCC {1,2}: reaches {1,2,3,4}, contributes 2*4 = 8
    // From SCC {3}: reaches {3,4}, contributes 1*2 = 2
    // From SCC {4}: reaches {4}, contributes 1*1 = 1
    // Total = 11
    EXPECT_EQ(rel.size(), 11);
}

}  // namespace test
}  // namespace souffle
