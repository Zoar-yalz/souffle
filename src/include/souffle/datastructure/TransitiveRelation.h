/*
 * Souffle - A Datalog Compiler
 * Copyright (c) 2025
 * Licensed under the Universal Permissive License v 1.0 as shown at:
 * - https://opensource.org/licenses/UPL
 * - <souffle root>/licenses/SOUFFLE-UPL.txt
 */

/************************************************************************
 *
 * @file TransitiveRelation.h
 *
 * A graph-backed binary relation exposing (reflexive) transitive reachability.
 *
 * Semantics (arity = 2):
 *   - insert(a,b) adds a directed edge a -> b
 *   - the relation contains (x,y) iff y is reachable from x via 0+ edges
 *     (i.e. reflexive-transitive closure over inserted edges)
 *
 * Notes:
 *   - This is designed to be usable in interpreter mode, hence it provides
 *     getBoundaries/lower_bound/upper_bound/partition with the same shape as
 *     other Souffle datastructures.
 *   - This implementation is intentionally simple: reachability is computed on
 *     demand (no incremental transitive-closure maintenance).
 *
 ***********************************************************************/

#pragma once

#include "souffle/RamTypes.h"
#include "souffle/datastructure/Graph.h"
#include "souffle/utility/ContainerUtil.h"
#include <roaring/roaring64map.hh>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace souffle {

template <typename TupleType,bool IncludeReflexive = true>
class TransitiveRelation {
public:
    using element_type = TupleType;
    using value_type = typename TupleType::value_type;


    /**
     * Operation hints for interpreter views.
     * We don't currently exploit temporal locality.
     */
    struct operation_hints {
        void clear() {}
    };
private:
    using GraphType = souffle::Graph<value_type>;
    using VertexSet = typename GraphType::VertexSet;

    GraphType g;
    bool dirty = true;

    // --- SCC cache (computed on demand) ---
    // Only keep the vertex -> SCC id mapping, plus the SCC DAG.
    mutable bool sccStale = true;
    mutable std::size_t sccCount = 0;
    mutable std::unordered_map<value_type, std::size_t> vertexToScc;
    // Reverse mapping: SCC id -> set of vertices in that SCC.
    mutable std::vector<VertexSet> sccToVertices;
    mutable souffle::Graph<std::size_t> sccDag;
    // SCC DAG in reverse topological order (sinks to sources).
    mutable std::vector<std::size_t> sccTopoRev;
    // SCC DAG reachability (transitive closure) for Purdom-style redundancy checks.
    // sccReach[u] is a roaring bitmap of SCC ids reachable from u via 1+ edges.
    mutable bool sccReachStale = true;
    //TODO add 32bit switch
    mutable std::vector<roaring::Roaring64Map> sccReach;
    // For non-reflexive mode: sccHasCycle[scc] = 1 if |SCC|>1 or vertex has self-loop
    // This means vertices in the SCC can reach themselves via 1+ edges.
    mutable std::vector<uint8_t> sccHasCycle;
    mutable std::size_t cachedSize = 0;
    mutable bool sizeStale = true;
    // Helpers
    bool vertexExists(const value_type v) const {
        return g.contains(v);
    }

    // Reachability check (respects IncludeReflexive template parameter).
    bool reachableOrSelf(const value_type from, const value_type to) const {
        if (!vertexExists(from) || !vertexExists(to)) {
            return false;
        }
        if (from == to) {
            if constexpr (IncludeReflexive) {
                return true;
            } else {
                // Non-reflexive: (x,x) only if x can reach itself via 1+ edges
                computeSccIfStale();
                const std::size_t scc = vertexToScc.at(from);
                return sccHasCycle[scc]!=0;
            }
        }
        return g.reaches(from, to);
    }

    VertexSet reachableSetFrom(const value_type from) const {
        if (!vertexExists(from)) {
            return {};
        }
        // Use SCC-based reachability if available.
        computeSccDagReachabilityIfStale();
        
        const std::size_t sccFrom = vertexToScc.at(from);
        VertexSet result;
        
        // Add all vertices in the source SCC.
        const auto& sccFromVerts = sccToVertices[sccFrom];
        if constexpr (IncludeReflexive) {
            // Include all vertices in source SCC (includes 'from' itself).
            result.insert(sccFromVerts.begin(), sccFromVerts.end());
        } else {
            // Non-reflexive: only include 'from' if SCC has a cycle
            if (sccHasCycle[sccFrom]) {
                result.insert(sccFromVerts.begin(), sccFromVerts.end());
            } else {
                // Single vertex SCC without self-loop: exclude 'from' itself
                for (const auto& v : sccFromVerts) {
                    if (v != from) {
                        result.insert(v);
                    }
                }
            }
        }
        
        // Add vertices from all reachable SCCs.
        const auto& reachableSccs = sccReach.at(sccFrom);
        for (auto it = reachableSccs.begin(); it != reachableSccs.end(); ++it) {
            const std::size_t scc = static_cast<std::size_t>(*it);
            const auto& verts = sccToVertices[scc];
            result.insert(verts.begin(), verts.end());
        }
        
        return result;
    }

    void computeSccIfStale() const {
        if (!sccStale) {
            return;
        }

        vertexToScc.clear();
        sccToVertices.clear();
        sccDag = souffle::Graph<std::size_t>{};
        sccCount = 0;
        sccTopoRev.clear();
        sccReachStale = true;
        sccReach.clear();
        sccHasCycle.clear();

        const auto& vertices = g.vertices();
        const std::size_t n = vertices.size();
        if (n == 0) {
            sccStale = false;
            return;
        }

        // Build vertex index mapping for O(1) access.
        // Map vertex -> index and index -> vertex.
        std::vector<value_type> indexToVertex;
        indexToVertex.reserve(n);
        std::unordered_map<value_type, std::size_t> vertexToIndex;
        vertexToIndex.reserve(n);
        
        for (const auto& v : vertices) {
            vertexToIndex[v] = indexToVertex.size();
            indexToVertex.push_back(v);
        }

        // Tarjan's SCC algorithm with explicit stack (non-recursive).
        const std::size_t UNVISITED = (std::size_t)-1;
        std::vector<std::size_t> index(n, UNVISITED);
        std::vector<std::size_t> lowlink(n);
        std::vector<bool> onStack(n, false);
        std::vector<std::size_t> sccId(n, UNVISITED);
        std::vector<std::size_t> S;  // Main stack
        S.reserve(n);
        
        std::size_t indexCounter = 0;
        std::size_t numSccs = 0;

        // DFS stack: (vertex_index, successor_iterator_position)
        struct DfsFrame {
            std::size_t v;
            std::size_t succIdx;
        };
        std::vector<DfsFrame> dfsStack;
        dfsStack.reserve(n);

        // Pre-build successor lists as indices for faster iteration.
        std::vector<std::vector<std::size_t>> succIndices(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& succs = g.successors(indexToVertex[i]);
            succIndices[i].reserve(succs.size());
            for (const auto& succ : succs) {
                auto it = vertexToIndex.find(succ);
                if (it != vertexToIndex.end()) {
                    succIndices[i].push_back(it->second);
                }
            }
        }

        for (std::size_t startIdx = 0; startIdx < n; ++startIdx) {
            if (index[startIdx] != UNVISITED) continue;

            // Start DFS from startIdx.
            dfsStack.push_back({startIdx, 0});
            
            while (!dfsStack.empty()) {
                auto& frame = dfsStack.back();
                const std::size_t v = frame.v;
                
                if (frame.succIdx == 0) {
                    // First visit to this node.
                    index[v] = indexCounter;
                    lowlink[v] = indexCounter;
                    ++indexCounter;
                    S.push_back(v);
                    onStack[v] = true;
                }

                // Process successors.
                const auto& succs = succIndices[v];
                bool pushedChild = false;
                
                while (frame.succIdx < succs.size()) {
                    const std::size_t w = succs[frame.succIdx];
                    ++frame.succIdx;
                    
                    if (index[w] == UNVISITED) {
                        // Recurse: push w onto DFS stack.
                        dfsStack.push_back({w, 0});
                        pushedChild = true;
                        break;
                    } else if (onStack[w]) {
                        lowlink[v] = std::min(lowlink[v], index[w]);
                    }
                }
                
                if (pushedChild) continue;

                // All successors processed; check if v is root of SCC.
                if (lowlink[v] == index[v]) {
                    // Pop SCC from stack.
                    sccToVertices.push_back(VertexSet{});
                    auto& currentSccVerts = sccToVertices.back();
                    
                    std::size_t w;
                    do {
                        w = S.back();
                        S.pop_back();
                        onStack[w] = false;
                        sccId[w] = numSccs;
                        currentSccVerts.insert(indexToVertex[w]);
                    } while (w != v);
                    
                    ++numSccs;
                }

                // Return from this node: update parent's lowlink.
                dfsStack.pop_back();
                if (!dfsStack.empty()) {
                    const std::size_t parent = dfsStack.back().v;
                    lowlink[parent] = std::min(lowlink[parent], lowlink[v]);
                }
            }
        }

        // Copy sccId to vertexToScc.
        sccCount = numSccs;
        for (std::size_t i = 0; i < n; ++i) {
            vertexToScc[indexToVertex[i]] = sccId[i];
        }

        // Build SCC condensation DAG.
        for (std::size_t scc = 0; scc < sccCount; ++scc) {
            sccDag.insert(scc);
        }
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t su = sccId[i];
            for (const std::size_t j : succIndices[i]) {
                const std::size_t sv = sccId[j];
                if (su != sv) {
                    sccDag.insert(su, sv);
                }
            }
        }

        // Compute reverse topological order for SCC DAG.
        std::vector<std::size_t> indegree(sccCount, 0);
        for (const auto& s : sccDag.vertices()) {
            for (const auto& t : sccDag.successors(s)) {
                ++indegree[t];
            }
        }

        std::set<std::size_t> ready;
        for (std::size_t i = 0; i < sccCount; ++i) {
            if (indegree[i] == 0) {
                ready.insert(i);
            }
        }

        std::vector<std::size_t> topo;
        topo.reserve(sccCount);
        while (!ready.empty()) {
            auto it = ready.begin();
            const std::size_t node = *it;
            ready.erase(it);

            topo.push_back(node);
            for (const auto& m : sccDag.successors(node)) {
                auto& d = indegree[m];
                if (--d == 0) {
                    ready.insert(m);
                }
            }
        }

        if (topo.size() != sccCount) {
            throw std::runtime_error("error: SCC DAG is not acyclic (unexpected)");
        }

        sccTopoRev.assign(topo.rbegin(), topo.rend());

        // Compute sccHasCycle: true(1) if |SCC|>1 or any vertex in SCC has self-loop
        sccHasCycle.resize(sccCount, 0);
        for (std::size_t scc = 0; scc < sccCount; ++scc) {
            const auto& verts = sccToVertices[scc];
            if (verts.size() > 1) {
                // Multiple vertices in SCC means there's a cycle
                sccHasCycle[scc] = 1;
            } else {
                // Single vertex: check for self-loop
                const auto& v = *verts.begin();
                const auto& succs = g.successors(v);
                sccHasCycle[scc] = (succs.find(v) != succs.end());
            }
        }

        sccReachStale = true;
        sccStale = false;
    }

    // Helper: merge reachability bitmaps (dst |= src)
    static inline void mergeReachBitmaps(roaring::Roaring64Map& dst, const roaring::Roaring64Map& src) {
        dst |= src;
    }

    // Helper: insert into roaring bitmap
    static inline void insertIntoBitmap(roaring::Roaring64Map& bm, std::size_t val) {
        bm.add(static_cast<uint64_t>(val));
    }

    void computeSccDagReachabilityIfStale() const {
        computeSccIfStale();
        if (!sccReachStale) {
            return;
        }

        // Initialize empty bitmaps for all SCCs.
        sccReach.resize(sccCount);
        for(std::size_t u = 0; u < sccCount; ++u) {
            sccReach[u].clear();
        }
        // Dynamic programming on DAG in reverse topological order (sinks -> sources).
        // reach[u] = union_{v in succ(u)} ( reach[v] U {v} )
        for (const auto& u : sccTopoRev) {
            auto& ru = sccReach[u];
            const auto& succs = sccDag.successors(u);
            if (succs.empty()) continue;
            
            for (const auto& v : succs) {
                mergeReachBitmaps(ru, sccReach[v]);
                insertIntoBitmap(ru, v);
            }
        }

        sccReachStale = false;
    }

public:
    TransitiveRelation() = default;

    /** Insert edge (a,b). Returns true if the underlying graph changed. */
    bool insert(value_type a, value_type b) {
        operation_hints h;
        return insert(a, b, h);
    }

    bool insert(value_type a, value_type b, operation_hints&) {
        bool changed = g.insert(a, b);
        if (changed) {
            dirty = true;
            sizeStale = true;
            // Mark SCC cache as stale; will be recomputed lazily on demand.
            sccStale = true;
            sccReachStale = true;
        }
        return changed;
    }

private:
    /**
     * Incrementally update SCC and reachability info after inserting edge a->b.
     * Strategy:
     *   - If both a and b are new vertices, create two new SCCs (or one if a==b).
     *   - If a->b crosses SCCs, may need to merge SCCs or update DAG reachability.
     *   - For simplicity, we only do cheap incremental updates; if the topology
     *     becomes complex (cycle introduced), we mark SCC as stale for full recompute.
     */
    void updateSccAndReachOnInsert(value_type a, value_type b) {
        // If SCC info was never computed, nothing to update.
        if (sccStale) {
            return;
        }

        bool aExisted = (vertexToScc.find(a) != vertexToScc.end());
        bool bExisted = (vertexToScc.find(b) != vertexToScc.end());

        if (!aExisted && !bExisted) {
            // Both are new vertices.
            if (a == b) {
                // Self-loop on new vertex: one new SCC.
                std::size_t newScc = sccCount++;
                vertexToScc[a] = newScc;
                sccToVertices.push_back(VertexSet{a});
                sccDag.insert(newScc);
                sccTopoRev.insert(sccTopoRev.begin(), newScc);
                if (!sccReachStale) {
                    sccReach.resize(sccCount);
                }
            } else {
                // Two new vertices, edge a->b.
                std::size_t sccA = sccCount++;
                std::size_t sccB = sccCount++;
                vertexToScc[a] = sccA;
                vertexToScc[b] = sccB;
                sccToVertices.push_back(VertexSet{a});
                sccToVertices.push_back(VertexSet{b});
                sccDag.insert(sccA, sccB);
                // Topo order: sccA before sccB (in forward order), reverse topo: sccB then sccA at front.
                sccTopoRev.insert(sccTopoRev.begin(), sccA);
                sccTopoRev.insert(sccTopoRev.begin(), sccB);
                if (!sccReachStale) {
                    sccReach.resize(sccCount);
                    sccReach[sccA].add(static_cast<uint64_t>(sccB));
                }
            }
            return;
        }

        if (!aExisted && bExisted) {
            // a is new, b exists.
            std::size_t sccB = vertexToScc[b];
            std::size_t sccA = sccCount++;
            vertexToScc[a] = sccA;
            sccToVertices.push_back(VertexSet{a});
            sccDag.insert(sccA, sccB);
            sccTopoRev.insert(sccTopoRev.begin(), sccA);
            if (!sccReachStale) {
                // sccA reaches sccB and everything sccB reaches.
                sccReach.resize(sccCount);
                sccReach[sccA] = sccReach[sccB];
                insertIntoBitmap(sccReach[sccA], sccB);
            }
            return;
        }

        if (aExisted && !bExisted) {
            // a exists, b is new.
            std::size_t sccA = vertexToScc[a];
            std::size_t sccB = sccCount++;
            vertexToScc[b] = sccB;
            sccToVertices.push_back(VertexSet{b});
            sccDag.insert(sccA, sccB);
            // b is a sink; insert at the front of reverse topo.
            sccTopoRev.insert(sccTopoRev.begin(), sccB);
            if (!sccReachStale) {
                sccReach.resize(sccCount);
                // Update reachability: sccA and all its predecessors can now reach sccB.
                updateReachabilityForNewSink(sccA, sccB);
            }
            return;
        }

        // Both a and b existed.
        std::size_t sccA = vertexToScc[a];
        std::size_t sccB = vertexToScc[b];

        if (sccA == sccB) {
            // Same SCC, no topology change.
            return;
        }

        // Check if this edge creates a cycle (b can reach a).
        // If so, SCCs need to be merged -> mark stale for full recompute.
        if (sccReaches(sccB, sccA) || sccB == sccA) {
            // Cycle introduced, must recompute SCCs.
            sccStale = true;
            return;
        }

        // No cycle: just add the DAG edge and update reachability.
        bool dagEdgeNew = sccDag.insert(sccA, sccB);
        if (dagEdgeNew && !sccReachStale) {
            // sccA and its predecessors now reach sccB and everything sccB reaches.
            propagateReachabilityFromEdge(sccA, sccB);
        }
    }

    /**
     * When a new sink SCC is added with edge from sccA to sccB (sccB is new sink),
     * update reachability for sccA and all its predecessors.
     */
    void updateReachabilityForNewSink(std::size_t sccA, std::size_t sccB) {
        // BFS/DFS backwards in the SCC DAG to find all predecessors of sccA.
        std::unordered_set<std::size_t> toUpdate;
        std::stack<std::size_t> stk;
        stk.push(sccA);
        while (!stk.empty()) {
            std::size_t cur = stk.top();
            stk.pop();
            if (toUpdate.count(cur)) continue;
            toUpdate.insert(cur);
            for (const auto& pred : sccDag.predecessors(cur)) {
                if (!toUpdate.count(pred)) {
                    stk.push(pred);
                }
            }
        }
        for (const auto& u : toUpdate) {
            insertIntoBitmap(sccReach[u], sccB);
        }
    }

    /**
     * Propagate reachability when a new DAG edge sccA->sccB is added.
     * sccA and all its predecessors now reach sccB and everything sccB reaches.
     */
    void propagateReachabilityFromEdge(std::size_t sccA, std::size_t sccB) {
        // Collect what sccB reaches (including sccB itself).
        roaring::Roaring64Map toAdd = sccReach[sccB];
        insertIntoBitmap(toAdd, sccB);

        // BFS/DFS backwards to find sccA and all its predecessors.
        std::unordered_set<std::size_t> toUpdate;
        std::stack<std::size_t> stk;
        stk.push(sccA);
        while (!stk.empty()) {
            std::size_t cur = stk.top();
            stk.pop();
            if (toUpdate.count(cur)) continue;
            toUpdate.insert(cur);
            for (const auto& pred : sccDag.predecessors(cur)) {
                if (!toUpdate.count(pred)) {
                    stk.push(pred);
                }
            }
        }
        for (const auto& u : toUpdate) {
            mergeReachBitmaps(sccReach[u], toAdd);
        }
    }

public:

    bool insert(const TupleType& t) {
        operation_hints h;
        return insert(t[0], t[1], h);
    }

    bool insert(const TupleType& t, operation_hints& h) {
        return insert(t[0], t[1], h);
    }

    /**
     * Contains in the (reflexive-)transitive closure.
     * Uses SCC-based reachability for efficiency when SCC info is available.
     * If IncludeReflexive=true: (x,x) is always reachable for any vertex x.
     * If IncludeReflexive=false: (x,x) is reachable only if x can reach itself via 1+ edges.
     */
    bool contains(value_type a, value_type b) const {
        if (!vertexExists(a) || !vertexExists(b)) {
            return false;
        }
        if (a == b) {
            if constexpr (IncludeReflexive) {
                return true;
            } else {
                // Non-reflexive: (x,x) only if x is in a cycle (|SCC|>1 or self-loop)
                computeSccIfStale();
                const std::size_t sccA = vertexToScc.at(a);
                return sccHasCycle[sccA]!=0;
            }
        }
        // Use SCC-based reachability if available.
        if (!sccStale) {
            std::size_t sccA = vertexToScc.at(a);
            std::size_t sccB = vertexToScc.at(b);
            if (sccA == sccB) {
                // Same SCC: a can reach b (within the SCC, a!=b guaranteed here).
                return true;
            }
            // Different SCCs: check if sccA can reach sccB in the DAG.
            return sccReaches(sccA, sccB);
        }
        // Fallback to BFS/DFS reachability.
        return g.reaches(a, b);
    }

    bool contains(const TupleType& t) const {
        return contains(t[0], t[1]);
    }

    bool contains(const TupleType& t, operation_hints&) const {
        return contains(t);
    }

    std::size_t size() const {
        if (sizeStale) {
            updateSize();
        }
        return cachedSize;
    }

    /**
     * Size of the closure (number of reachable pairs).
     * Optimized: for each SCC, contribution = |SCC nodes| * |reachable nodes from SCC|.
     * In non-reflexive mode, subtract self-pairs that are not in cycles.
     */
    void updateSize() const {
        computeSccDagReachabilityIfStale();
        
        // Precompute total reachable vertices for each SCC.
        // totalReach[scc] = |vertices in scc| + sum(|vertices in each reachable scc|)
        std::vector<std::size_t> totalReach(sccCount, 0);
        for (std::size_t scc = 0; scc < sccCount; ++scc) {
            totalReach[scc] = sccToVertices[scc].size();
            const auto& reachableSccs = sccReach.at(scc);
            for (auto it = reachableSccs.begin(); it != reachableSccs.end(); ++it) {
                const std::size_t reachScc = static_cast<std::size_t>(*it);
                totalReach[scc] += sccToVertices[reachScc].size();
            }
        }
        
        // Sum: for each SCC, |SCC nodes| * |reachable nodes from SCC|.
        std::size_t total = 0;
        for (std::size_t scc = 0; scc < sccCount; ++scc) {
            const std::size_t sccSize = sccToVertices[scc].size();
            total += sccSize * totalReach[scc];
            
            // In non-reflexive mode, subtract self-pairs for SCCs without cycles
            if constexpr (!IncludeReflexive) {
                if (sccHasCycle[scc]==0) {
                    // Each vertex in this SCC cannot reach itself
                    total -= sccSize;
                }
            }
        }
        cachedSize = total;
        sizeStale = false;
    }

    bool empty() const {
        return g.vertices().empty();
    }

    void clear() {
        g = GraphType{};
        dirty = true;
        sccStale = true;
        vertexToScc.clear();
        sccDag = souffle::Graph<std::size_t>{};
        sccCount = 0;
        sccTopoRev.clear();
        sccReachStale = true;
        sccReach.clear();
        sccHasCycle.clear();
        cachedSize = 0;
        sizeStale = true;
    }

    /**
     * Compute SCCs of the underlying edge graph (if stale) and cache the mappings.
     *
     * SCC ids are assigned in discovery order of Gabow's algorithm.
     */
    void computeScc() const {
        computeSccIfStale();
    }

    /** Number of SCCs (0 if graph has no vertices). */
    std::size_t getNumberOfSccs() const {
        computeSccIfStale();
        return sccCount;
    }

    /** Get SCC id for an existing vertex. Throws if vertex is not present. */
    std::size_t getSccOf(const value_type v) const {
        computeSccIfStale();
        auto it = vertexToScc.find(v);
        if (it == vertexToScc.end()) {
            throw std::out_of_range("error: getSccOf() called with a vertex not present in the graph");
        }
        return it->second;
    }

    /**
     * Get the SCC condensation graph (a DAG) of the underlying edge graph.
     * Nodes are SCC ids in the range [0, getNumberOfSccs()).
     */
    const souffle::Graph<std::size_t>& buildSccDag() const {
        computeSccIfStale();
        return sccDag;
    }

    /**
     * SCC DAG in reverse topological order (sinks to sources).
     * Suitable for reverse-strata evaluation.
     */
    const std::vector<std::size_t>& sccTopologicalReverseOrder() const {
        computeSccIfStale();
        return sccTopoRev;
    }

    /** Get all vertices in a specific SCC. */
    const VertexSet& getVerticesInScc(std::size_t sccId) const {
        computeSccIfStale();
        return sccToVertices[sccId];
    }

    /** Print the reverse topological order as a whitespace-separated list of SCC ids. */
    void printSccTopologicalReverseOrder(std::ostream& os) const {
        const auto& xs = sccTopologicalReverseOrder();
        for (std::size_t i = 0; i < xs.size(); ++i) {
            if (i != 0) os << ' ';
            os << xs[i];
        }
    }

    /**
     * Compute and cache SCC DAG reachability (transitive closure) for Purdom.
     *
     * After calling this, sccReaches(u, v) answers whether SCC v is reachable from SCC u.
     */
    void computeSccDagReachability() const {
        computeSccDagReachabilityIfStale();
    }

    /** Returns true if SCC `to` is reachable from SCC `from` via 1+ edges in the SCC DAG. */
    bool sccReaches(const std::size_t from, const std::size_t to) const {
        computeSccDagReachabilityIfStale();
        if (from >= sccReach.size()) return false;
        const auto& bm = sccReach[from];
        return bm.contains(static_cast<uint64_t>(to));
    }

    /** Get the full reachability bitmap for an SCC. */
    const roaring::Roaring64Map& getSccReachBitmap(const std::size_t scc) const {
        computeSccDagReachabilityIfStale();
        static const roaring::Roaring64Map emptyBitmap;
        if (scc >= sccReach.size()) return emptyBitmap;
        return sccReach[scc];
    }

    /**
     * Iterator over the closure tuples.
     * Enumerates pairs in stable vertex/set order.
     */
    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = TupleType;
        using difference_type = ptrdiff_t;
        using pointer = const value_type*;
        using reference = const value_type&;
        // Vertex type (element of tuple)
        using vertex_type = typename TupleType::value_type;

        explicit iterator(const TransitiveRelation* rel, bool /*endTag*/)
                : rel(rel), isEndVal(true) {}

        // ALL: iterate all sources, then all reachable targets.
        explicit iterator(const TransitiveRelation* rel) : rel(rel), mode(Mode::All) {
            fromIt  = rel->g.vertices().begin();
            fromEnd = rel->g.vertices().end();

            while (fromIt != fromEnd) {
                if (initForSource(*fromIt)) {
                    isEndVal = false;
                    return;
                }
                ++fromIt;  //skip to next scc
            }
            isEndVal = true; 
        }

        // FROM: iterate reachable targets for a fixed source.
        explicit iterator(const TransitiveRelation* rel, vertex_type from)
                : rel(rel), mode(Mode::From), fixedFrom(from) {
            if (!rel->vertexExists(from)) {
                isEndVal = true;
                return;
            }
            if (!initForSource(from)) {
                isEndVal = true;   // 固定 source 本身无输出 => 迭代为空
                return;
            }        
        }

        // SINGLE: exactly one tuple.
        explicit iterator(const TransitiveRelation* rel, vertex_type from, vertex_type to)
                : rel(rel), mode(Mode::Single), fixedFrom(from), fixedTo(to) {
            if (!rel->reachableOrSelf(from, to)) {
                isEndVal = true;
                return;
            }
            current[0] = from;
            current[1] = to;
            singleDone = false;
        }

        iterator(const iterator&) = default;
        iterator(iterator&&) = default;
        iterator& operator=(const iterator&) = default;

        bool operator==(const iterator& other) const {
            if (isEndVal && other.isEndVal) return rel == other.rel;
            return isEndVal == other.isEndVal && current == other.current;
        }

        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }

        reference operator*() const {
            return current;
        }

        pointer operator->() const {
            return &current;
        }

        iterator& operator++() {
            if (isEndVal) {
                throw std::out_of_range("error: incrementing an out of range iterator");
            }

            switch (mode) {
                case Mode::Single:
                    if (singleDone) {
                        isEndVal = true;
                        return *this;
                    }
                    singleDone = true;
                    isEndVal = true;
                    return *this;

                case Mode::From:
                    advanceWithinReachable();
                    return *this;

                case Mode::All:
                    // Advance within current reachable set first.
                    if (advanceWithinReachableNoEnd()) {
                        return *this;
                    }
                    do {
                        ++fromIt;
                        if (fromIt == fromEnd) {
                            isEndVal = true;
                            return *this;
                        }
                    } while (!initForSource(*fromIt));
                    return *this;
            }

            return *this;
        }

    private:
        enum class Mode { All, From, Single };

        const TransitiveRelation* rel = nullptr;
        bool isEndVal = false;
        Mode mode = Mode::All;

        TupleType current{};

        // ALL mode
        typename VertexSet::const_iterator fromIt{};
        typename VertexSet::const_iterator fromEnd{};

        // FROM/ALL mode: current source and its reachable-set.
        // Instead of copying, we iterate directly through SCC structures.
        vertex_type curFrom{};
        std::size_t curFromScc{};
        
        // SCCs to iterate: first the source SCC, then reachable SCCs.
        // We use a vector to hold the sequence of SCCs to visit.
        std::vector<std::size_t> sccSequence;
        std::size_t sccSeqIdx = 0;
        
        // Iterator within current SCC's vertex set.
        typename VertexSet::const_iterator vertIt{};
        typename VertexSet::const_iterator vertEnd{};

        // SINGLE mode
        vertex_type fixedFrom{};
        vertex_type fixedTo{};
        bool singleDone = true;

        bool initForSource(const vertex_type from) {
            curFrom = from;
            
            // Ensure SCC info is computed.
            rel->computeSccDagReachabilityIfStale();
            
            curFromScc = rel->vertexToScc.at(from);
            
            // Build sequence of SCCs to visit: source SCC first, then all reachable SCCs.
            sccSequence.clear();
            sccSequence.push_back(curFromScc);
            const auto& reachableSccs = rel->sccReach.at(curFromScc);
            for (auto it = reachableSccs.begin(); it != reachableSccs.end(); ++it) {
                sccSequence.push_back(static_cast<std::size_t>(*it));
            }
            
            sccSeqIdx = 0;
            
            // Start iterating within the first SCC.
            const auto& firstSccVerts = rel->sccToVertices[sccSequence[0]];
            vertIt = firstSccVerts.begin();
            vertEnd = firstSccVerts.end();
            
            if (vertIt == vertEnd) {
                // Should not happen for non-empty graph.
                isEndVal = true;
                return false;
            }

            current[0] = curFrom;
            current[1] = *vertIt;
            
            // In non-reflexive mode, skip self if the SCC has no cycle.
            if constexpr (!IncludeReflexive) {
                if (!skipSelfIfNeeded()) return false; 
            }
            return true;
        }
        
        // Skip the current element if it's a self-pair and we're in non-reflexive mode
        // and the SCC has no cycle (i.e., it's a singleton without self-loop).
        bool skipSelfIfNeeded() {
            if constexpr (!IncludeReflexive) {
                while (!isEndVal && current[0] == current[1]) {
                    // Check if this SCC has a real cycle.
                    std::size_t curScc = sccSequence[sccSeqIdx];
                    if (rel->sccHasCycle[curScc]==0) {
                        // No cycle, so skip this self-pair.
                        if (!advanceWithinReachableNoEnd()) {
                            return false;
                        }
                    } else {
                        // SCC has a cycle, self-pair is valid.
                        break;
                    }
                }
            }
            return true;
        }

        bool advanceWithinReachableNoEnd() {
            ++vertIt;
            
            // If finished current SCC, move to next SCC in sequence.
            while (vertIt == vertEnd) {
                ++sccSeqIdx;
                if (sccSeqIdx >= sccSequence.size()) {
                    return false;
                }
                const auto& nextSccVerts = rel->sccToVertices[sccSequence[sccSeqIdx]];
                vertIt = nextSccVerts.begin();
                vertEnd = nextSccVerts.end();
            }
            
            current[1] = *vertIt;
            
            // In non-reflexive mode, if this is a self-pair and the SCC has no cycle, skip it.
            if constexpr (!IncludeReflexive) {
                while (current[0] == current[1]) {
                    std::size_t curScc = sccSequence[sccSeqIdx];
                    if (rel->sccHasCycle[curScc]==0) {
                        // Skip this self-pair.
                        ++vertIt;
                        while (vertIt == vertEnd) {
                            ++sccSeqIdx;
                            if (sccSeqIdx >= sccSequence.size()) {
                                return false;
                            }
                            const auto& nextSccVerts = rel->sccToVertices[sccSequence[sccSeqIdx]];
                            vertIt = nextSccVerts.begin();
                            vertEnd = nextSccVerts.end();
                        }
                        current[1] = *vertIt;
                    } else {
                        break;
                    }
                }
            }
            
            return true;
        }

        void advanceWithinReachable() {
            if (!advanceWithinReachableNoEnd()) {
                isEndVal = true;
            }
        }
    };

    iterator begin() const {
        return iterator(this);
    }

    iterator end() const {
        return iterator(this, true);
    }

    /**
     * Prefix query API used by generated/interpreter indexes.
     */
    template <unsigned levels>
    souffle::range<iterator> getBoundaries(const TupleType& entry) const {
        operation_hints h;
        return getBoundaries<levels>(entry, h);
    }

    template <unsigned levels>
    souffle::range<iterator> getBoundaries(const TupleType& entry, operation_hints&) const {
        if constexpr (levels == 0) {
            return make_range(begin(), end());
        }

        if constexpr (levels == 1) {
            if (!vertexExists(entry[0])) return make_range(end(), end());
            return make_range(iterator(this, entry[0]), end());
        }

        if constexpr (levels == 2) {
            if (!contains(entry[0], entry[1])) return make_range(end(), end());
            return make_range(iterator(this, entry[0], entry[1]), end());
        }

        static_assert(levels <= 2, "TransitiveRelation supports only arity=2");
        return make_range(end(), end());
    }

    /**
     * InterpreterIndex compatibility.
     * Uses MIN_RAM_SIGNED as the wildcard for unbound fields.
     */
    iterator lower_bound(const TupleType& entry, operation_hints&) const {
        if constexpr (std::is_same_v<value_type, RamDomain>) {
            if (entry[0] == MIN_RAM_SIGNED && entry[1] == MIN_RAM_SIGNED) {
                return begin();
            }
            if (entry[0] != MIN_RAM_SIGNED && entry[1] == MIN_RAM_SIGNED) {
                if (!vertexExists(entry[0])) return end();
                return iterator(this, entry[0]);
            }
            if (entry[0] != MIN_RAM_SIGNED && entry[1] != MIN_RAM_SIGNED) {
                if (!contains(entry[0], entry[1])) return end();
                return iterator(this, entry[0], entry[1]);
            }
            return end();
        } else {
            // Fallback for non-RamDomain TupleType (no wildcard semantics).
            if (!contains(entry)) return end();
            return iterator(this, entry[0], entry[1]);
        }
    }

    iterator lower_bound(const TupleType& entry) const {
        operation_hints h;
        return lower_bound(entry, h);
    }

    iterator upper_bound(const TupleType&, operation_hints&) const {
        return end();
    }

    iterator upper_bound(const TupleType& entry) const {
        operation_hints h;
        return upper_bound(entry, h);
    }

    /**
     * Partition for approximate parallel iteration.
     * Simple strategy: one chunk per source vertex (may exceed requested chunks).
     */
    std::vector<souffle::range<iterator>> partition(std::size_t chunks) const {
        (void)chunks;
        std::vector<souffle::range<iterator>> out;
        if (g.vertices().empty()) return out;

        // keep it simple: yield one range per source vertex
        for (const auto& from : g.vertices()) {
            out.push_back(souffle::make_range(iterator(this, from), end()));
        }
        return out;
    }

    void printStats(std::ostream& /*o*/) const {}
};

}  // namespace souffle
