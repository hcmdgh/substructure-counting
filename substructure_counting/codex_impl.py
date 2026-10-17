"""Per-node counts of undirected subgraphs with three to five vertices.

Non-induced occurrences are distinct *edge subgraphs*, modulo automorphisms
of the requested pattern.  For example, K4 contains three 4-cycles, six
 diamonds, and four 3-stars.  Induced occurrences are vertex sets instead.
The implementation uses sorted CSR adjacency, never a dense n-by-n matrix.
"""

from functools import lru_cache
from itertools import permutations
from numbers import Integral

import numpy as np
import torch
from numba import njit
from torch import Tensor


# Vertex orders keep prefixes connected and close cycles early.  In particular,
# centers of stars/wheels/bow-ties and the degree-three diamond vertices come first.
_PATTERNS = {
    "triangle": (3, ((0, 1), (0, 2), (1, 2)), (0, 1, 2)),
    "3-star": (4, ((0, 1), (0, 2), (0, 3)), (0, 1, 2, 3)),
    "4-cycle": (4, ((0, 1), (1, 2), (2, 3), (3, 0)), (0, 1, 3, 2)),
    "diamond": (4, ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)), (0, 2, 1, 3)),
    "4-clique": (4, tuple((u, v) for u in range(4) for v in range(u + 1, 4)), (0, 1, 2, 3)),
    "4-star": (5, ((0, 1), (0, 2), (0, 3), (0, 4)), (0, 1, 2, 3, 4)),
    "5-cycle": (5, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)), (0, 1, 4, 2, 3)),
    "banner": (5, ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4)), (0, 1, 3, 2, 4)),
    "bow-tie": (5, ((0, 1), (0, 2), (1, 2), (0, 3), (3, 4), (4, 0)), (0, 1, 2, 3, 4)),
    "biclique": (5, ((0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)), (0, 2, 1, 3, 4)),
    "wheel": (5, ((0, 1), (1, 2), (2, 3), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)), (4, 0, 1, 3, 2)),
    "5-clique": (5, tuple((u, v) for u in range(5) for v in range(u + 1, 5)), (0, 1, 2, 3, 4)),
}


@lru_cache(maxsize=len(_PATTERNS))
def _pattern_data(name):
    """Build symmetry constraints from a stabilizer chain (at most 5! work)."""
    size, edges, order = _PATTERNS[name]
    adjacency = np.zeros((size, size), dtype=np.bool_)
    positions = {vertex: index for index, vertex in enumerate(order)}
    for u, v in edges:
        u, v = positions[u], positions[v]
        adjacency[u, v] = adjacency[v, u] = True
    automorphisms = [
        p for p in permutations(range(size))
        if all(adjacency[i, j] == adjacency[p[i], p[j]]
               for i in range(size) for j in range(i))
    ]
    # At each depth choose the smallest host vertex in the current orbit, then
    # fix that pattern vertex.  This gives exactly one representative per copy,
    # including when extra host edges create additional pattern embeddings.
    before = np.zeros((size, size), dtype=np.bool_)
    for i in range(size):
        for p in automorphisms:
            if p[i] != i:
                before[i, p[i]] = True
        automorphisms = [p for p in automorphisms if p[i] == i]
    return adjacency, before, adjacency.sum(axis=1).astype(np.int64)


@njit(inline="always")
def _lower_bound(indices, start, end, value):
    while start < end:
        mid = (start + end) // 2
        if indices[mid] < value:
            start = mid + 1
        else:
            end = mid
    return start


@njit(inline="always")
def _adjacent(indptr, indices, u, v):
    # Binary search the shorter of the two sorted neighborhoods.
    if indptr[u + 1] - indptr[u] > indptr[v + 1] - indptr[v]:
        u, v = v, u
    end = indptr[u + 1]
    pos = _lower_bound(indices, indptr[u], end, v)
    return pos < end and indices[pos] == v


@njit(inline="always")
def _choose(n, k):
    # k is at most four.  Divide out factors before multiplying to avoid
    # intermediate overflow when the final binomial coefficient fits int64.
    if n < k or k < 0:
        return np.int64(0)
    result = np.int64(1)
    for i in range(1, k + 1):
        factor = n - i + 1
        denominator = i
        a, b = factor, denominator
        while b:
            a, b = b, a % b
        factor //= a
        denominator //= a
        result = (result // denominator) * factor
    return result


@njit
def _count_stars(indptr, indices, leaves, induced):
    n = len(indptr) - 1
    result = np.zeros(n, dtype=np.int64)
    for center in range(n):
        start, end = indptr[center], indptr[center + 1]
        degree = end - start
        if degree < leaves:
            continue
        independent = not induced
        if induced:
            independent = True
            # An independent neighborhood admits a binomial formula, critical
            # for hubs in trees and triangle-free graphs.
            for pos in range(start, end):
                u = indices[pos]
                # Scan the shorter list, so a high-degree star costs O(m).
                if indptr[u + 1] - indptr[u] < degree:
                    for q in range(indptr[u], indptr[u + 1]):
                        v = indices[q]
                        if v > u and _adjacent(indptr, indices, center, v):
                            independent = False
                            break
                else:
                    for q in range(pos + 1, end):
                        if _adjacent(indptr, indices, u, indices[q]):
                            independent = False
                            break
                if not independent:
                    break
        if independent:
            result[center] += _choose(degree, leaves)
            per_leaf = _choose(degree - 1, leaves - 1)
            for pos in range(start, end):
                result[indices[pos]] += per_leaf
        else:
            # Enumerate independent leaf sets in increasing CSR order.  The
            # final leaf is aggregated for the center and previously chosen leaves.
            chosen = np.empty(4, dtype=np.int64)
            cursor = np.empty(4, dtype=np.int64)
            cursor[0] = start
            depth = 0
            while depth >= 0:
                limit = end - (leaves - depth - 1)
                if cursor[depth] >= limit:
                    depth -= 1
                    continue
                pos = cursor[depth]
                cursor[depth] += 1
                vertex = indices[pos]
                valid = True
                for j in range(depth):
                    if _adjacent(indptr, indices, vertex, chosen[j]):
                        valid = False
                        break
                if not valid:
                    continue
                chosen[depth] = vertex
                if depth == leaves - 2:
                    total = np.int64(0)
                    for q in range(pos + 1, end):
                        last = indices[q]
                        valid = True
                        for j in range(depth + 1):
                            if _adjacent(indptr, indices, last, chosen[j]):
                                valid = False
                                break
                        if valid:
                            result[last] += 1
                            total += 1
                    result[center] += total
                    for j in range(depth + 1):
                        result[chosen[j]] += total
                else:
                    depth += 1
                    cursor[depth] = pos + 1
    return result


@njit
def _count_embeddings(indptr, indices, adjacency, before, min_degree, induced):
    """Sparse, symmetry-broken subgraph matching; no n-choose-k subset scan."""
    n = len(indptr) - 1
    size = len(adjacency)
    result = np.zeros(n, dtype=np.int64)
    mapped = np.empty(size, dtype=np.int64)
    cursor = np.empty(size, dtype=np.int64)
    ends = np.empty(size, dtype=np.int64)
    for root in range(n):
        if indptr[root + 1] - indptr[root] < min_degree[0]:
            continue
        mapped[0] = root
        depth = 1
        initialize = True
        while depth > 0:
            if initialize:
                # Intersect implicitly by scanning the smallest already mapped
                # required neighborhood and checking the other constraints.
                anchor = -1
                best_degree = n + 1
                lower = 0
                for j in range(depth):
                    if before[j, depth]:
                        lower = max(lower, mapped[j] + 1)
                    if adjacency[j, depth]:
                        d = indptr[mapped[j] + 1] - indptr[mapped[j]]
                        if d < best_degree:
                            anchor, best_degree = mapped[j], d
                ends[depth] = indptr[anchor + 1]
                cursor[depth] = _lower_bound(indices, indptr[anchor], ends[depth], lower)
                initialize = False
            if cursor[depth] == ends[depth]:
                depth -= 1
                continue
            vertex = indices[cursor[depth]]
            cursor[depth] += 1
            if indptr[vertex + 1] - indptr[vertex] < min_degree[depth]:
                continue
            valid = True
            for j in range(depth):
                if mapped[j] == vertex:
                    valid = False
                    break
                edge = _adjacent(indptr, indices, vertex, mapped[j])
                if adjacency[j, depth]:
                    if not edge:
                        valid = False
                        break
                elif induced and edge:
                    valid = False
                    break
            if not valid:
                continue
            mapped[depth] = vertex
            if depth == size - 1:
                for j in range(size):
                    result[mapped[j]] += 1
            else:
                depth += 1
                initialize = True
    return result


@njit
def _count_common_neighbor_patterns(indptr, indices, biclique):
    """Non-induced diamond/K2,3 counts via binomial common-neighbor counts."""
    n = len(indptr) - 1
    result = np.zeros(n, dtype=np.int64)
    common = np.empty(n, dtype=np.int64)
    seen = np.full(n, -1, dtype=np.int64)
    candidates = np.empty(n, dtype=np.int64)
    for u in range(n):
        if biclique:
            if indptr[u + 1] - indptr[u] < 3:
                continue
            count = 0
            # Only distance-two hub pairs can have common neighbors.  A timestamp
            # array deduplicates pairs without storing an n-by-n matrix.
            for pos in range(indptr[u], indptr[u + 1]):
                w = indices[pos]
                if indptr[w + 1] - indptr[w] < 2:
                    continue
                for q in range(indptr[w], indptr[w + 1]):
                    v = indices[q]
                    if v > u and seen[v] != u and indptr[v + 1] - indptr[v] >= 3:
                        seen[v] = u
                        candidates[count] = v
                        count += 1
        else:
            if indptr[u + 1] - indptr[u] < 3:
                continue
            count = 0
            for pos in range(indptr[u], indptr[u + 1]):
                v = indices[pos]
                if v > u and indptr[v + 1] - indptr[v] >= 3:
                    candidates[count] = v
                    count += 1
        for pair in range(count):
            v = candidates[pair]
            a, a_end = indptr[u], indptr[u + 1]
            b, b_end = indptr[v], indptr[v + 1]
            total = 0
            while a < a_end and b < b_end:
                if indices[a] < indices[b]:
                    a += 1
                elif indices[b] < indices[a]:
                    b += 1
                else:
                    common[total] = indices[a]
                    total += 1
                    a += 1
                    b += 1
            leaves = 3 if biclique else 2
            copies = _choose(total, leaves)
            if copies:
                result[u] += copies
                result[v] += copies
                per_leaf = _choose(total - 1, leaves - 1)
                for j in range(total):
                    result[common[j]] += per_leaf
    return result
