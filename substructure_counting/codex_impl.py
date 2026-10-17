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
    if not induced:
        for center in range(n):
            degree = indptr[center + 1] - indptr[center]
            if degree >= leaves:
                result[center] += _choose(degree, leaves)
                per_leaf = _choose(degree - 1, leaves - 1)
                for pos in range(indptr[center], indptr[center + 1]):
                    result[indices[pos]] += per_leaf
        return result

    membership = np.full(n, -1, dtype=np.int64)
    active = np.empty(n, dtype=np.int64)
    selected = np.empty(4, dtype=np.int64)
    cursor = np.empty(4, dtype=np.int64)
    for center in range(n):
        start, end = indptr[center], indptr[center + 1]
        degree = end - start
        if degree < leaves:
            continue
        for pos in range(start, end):
            membership[indices[pos]] = center
        count = 0
        matching = True
        # Only leaves incident to a triangle at the center need enumeration.
        # All other leaves are independent of the entire neighborhood and can
        # be attached in bulk using binomial coefficients.  In particular, one
        # triangle at a hub must not cause enumeration of C(degree, 4) stars.
        for pos in range(start, end):
            u = indices[pos]
            internal = 0
            if indptr[u + 1] - indptr[u] <= degree:
                for q in range(indptr[u], indptr[u + 1]):
                    if membership[indices[q]] == center:
                        internal += 1
                        if internal == 2:
                            break
            else:
                for q in range(start, end):
                    if _adjacent(indptr, indices, u, indices[q]):
                        internal += 1
                        if internal == 2:
                            break
            if internal:
                active[count] = u
                count += 1
            if internal == 2:
                matching = False
        isolated = degree - count
        if matching:
            # If internal edges form a matching, choose at most one endpoint
            # per edge and choose the remaining leaves among isolated vertices.
            pairs = count // 2
            copies = np.int64(0)
            per_active = np.int64(0)
            per_isolated = np.int64(0)
            for j in range(leaves + 1):
                copies += _choose(pairs, j) * (1 << j) * _choose(isolated, leaves - j)
                per_active += _choose(pairs - 1, j) * (1 << j) * _choose(isolated, leaves - 1 - j)
                per_isolated += _choose(pairs, j) * (1 << j) * _choose(isolated - 1, leaves - 1 - j)
            result[center] += copies
            j = 0
            for pos in range(start, end):
                vertex = indices[pos]
                if j < count and active[j] == vertex:
                    result[vertex] += per_active
                    j += 1
                else:
                    result[vertex] += per_isolated
            continue
        result[center] += _choose(isolated, leaves)
        per_isolated = _choose(isolated - 1, leaves - 1)
        if count:
            depth = 0
            cursor[0] = 0
            while depth >= 0:
                if cursor[depth] >= count:
                    depth -= 1
                    continue
                pos = cursor[depth]
                cursor[depth] += 1
                # Insufficient remaining active/isolated leaves.
                if count - pos + isolated < leaves - depth:
                    depth -= 1
                    continue
                vertex = active[pos]
                valid = True
                for j in range(depth):
                    if _adjacent(indptr, indices, vertex, selected[j]):
                        valid = False
                        break
                if not valid:
                    continue
                selected[depth] = vertex
                remaining = leaves - depth - 1
                weight = _choose(isolated, remaining)
                result[center] += weight
                for j in range(depth + 1):
                    result[selected[j]] += weight
                per_isolated += _choose(isolated - 1, remaining - 1)
                if remaining:
                    depth += 1
                    cursor[depth] = pos + 1
        if per_isolated:
            # The active list is sorted, as is the original neighborhood.
            j = 0
            for pos in range(start, end):
                vertex = indices[pos]
                if j < count and active[j] == vertex:
                    j += 1
                else:
                    result[vertex] += per_isolated
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


@njit
def _forward_csr(indptr, indices):
    """Orient from smaller (degree, node ID) to larger, bounding forward degree."""
    n = len(indptr) - 1
    forward_ptr = np.empty(n + 1, dtype=np.int64)
    forward = np.empty(len(indices) // 2, dtype=np.int64)
    used = 0
    forward_ptr[0] = 0
    for u in range(n):
        degree = indptr[u + 1] - indptr[u]
        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]
            other = indptr[v + 1] - indptr[v]
            if degree < other or (degree == other and u < v):
                forward[used] = v
                used += 1
        forward_ptr[u + 1] = used
    return forward_ptr, forward


@njit
def _count_cliques(indptr, indices, size):
    """Intersect forward candidate lists; count final extensions in bulk."""
    n = len(indptr) - 1
    result = np.zeros(n, dtype=np.int64)
    max_degree = 0
    for u in range(n):
        max_degree = max(max_degree, indptr[u + 1] - indptr[u])
    if max_degree < size - 1:
        return result
    candidates = np.empty((size, max_degree), dtype=np.int64)
    lengths = np.empty(size, dtype=np.int64)
    cursor = np.empty(size, dtype=np.int64)
    mapped = np.empty(size, dtype=np.int64)
    for root in range(n):
        count = indptr[root + 1] - indptr[root]
        if count < size - 1:
            continue
        mapped[0] = root
        for p in range(count):
            candidates[0, p] = indices[indptr[root] + p]
        lengths[0] = count
        cursor[0] = 0
        depth = 0
        while depth >= 0:
            if cursor[depth] == lengths[depth]:
                depth -= 1
                continue
            vertex = candidates[depth, cursor[depth]]
            cursor[depth] += 1
            mapped[depth + 1] = vertex
            a = 0
            b, end = indptr[vertex], indptr[vertex + 1]
            count = 0
            while a < lengths[depth] and b < end:
                x, y = candidates[depth, a], indices[b]
                if x < y:
                    a += 1
                elif y < x:
                    b += 1
                else:
                    candidates[depth + 1, count] = x
                    count += 1
                    a += 1
                    b += 1
            if depth + 2 == size - 1:
                for j in range(depth + 2):
                    result[mapped[j]] += count
                for j in range(count):
                    result[candidates[depth + 1, j]] += 1
            elif count >= size - depth - 2:
                depth += 1
                lengths[depth] = count
                cursor[depth] = 0
    return result


@njit
def _count_four_cycles(indptr, indices, induced, banner):
    """Aggregate common-neighbor pairs instead of enumerating each 4-cycle.

    Every cycle is assigned to its smallest vertex u and the opposite vertex v.
    For banners, cycle-node counts are degree weighted and pendant-node counts
    are obtained by a final neighbor sum, avoiding one loop per pendant edge.
    """
    n = len(indptr) - 1
    result = np.zeros(n, dtype=np.int64)
    cycle_counts = np.zeros(n, dtype=np.int64)
    seen = np.full(n, -1, dtype=np.int64)
    candidates = np.empty(n, dtype=np.int64)
    common = np.empty(n, dtype=np.int64)
    member = np.zeros(n, dtype=np.bool_)
    internal_degree = np.zeros(n, dtype=np.int64)
    for u in range(n):
        if indptr[u + 1] - indptr[u] < 2:
            continue
        count = 0
        for pos in range(indptr[u], indptr[u + 1]):
            w = indices[pos]
            if w <= u or indptr[w + 1] - indptr[w] < 2:
                continue
            for q in range(indptr[w], indptr[w + 1]):
                v = indices[q]
                if v > u and seen[v] != u and indptr[v + 1] - indptr[v] >= 2:
                    seen[v] = u
                    candidates[count] = v
                    count += 1
        for pair in range(count):
            v = candidates[pair]
            chord = _adjacent(indptr, indices, u, v) if induced or banner else False
            if induced and chord:
                continue
            a = _lower_bound(indices, indptr[u], indptr[u + 1], u + 1)
            b = _lower_bound(indices, indptr[v], indptr[v + 1], u + 1)
            a_end, b_end = indptr[u + 1], indptr[v + 1]
            total = 0
            degree_sum = np.int64(0)
            while a < a_end and b < b_end:
                x, y = indices[a], indices[b]
                if x < y:
                    a += 1
                elif y < x:
                    b += 1
                else:
                    common[total] = x
                    total += 1
                    degree_sum += indptr[x + 1] - indptr[x]
                    a += 1
                    b += 1
            if total < 2:
                continue
            internal_edges = np.int64(0)
            if induced or banner:
                for j in range(total):
                    member[common[j]] = True
                    internal_degree[common[j]] = 0
                # Use the shorter of a common list and a full neighborhood.
                for j in range(total):
                    x = common[j]
                    if indptr[x + 1] - indptr[x] < total:
                        for q in range(indptr[x], indptr[x + 1]):
                            y = indices[q]
                            if y > x and member[y]:
                                internal_degree[x] += 1
                                internal_degree[y] += 1
                                internal_edges += 1
                    else:
                        for k in range(j + 1, total):
                            y = common[k]
                            if _adjacent(indptr, indices, x, y):
                                internal_degree[x] += 1
                                internal_degree[y] += 1
                                internal_edges += 1
                for j in range(total):
                    member[common[j]] = False
            copies = _choose(total, 2)
            if banner:
                # W(C) = sum(degrees of cycle nodes) - 8 - 2*diagonals.
                # Subtract internal pendant choices here; add all choices via
                # sum_{neighbor c} cycle_counts[c] after processing the cycles.
                base = indptr[u + 1] - indptr[u] + indptr[v + 1] - indptr[v] - 8 - 2 * chord
                hubs = (base - 2 - chord) * copies + (total - 1) * degree_sum - 2 * internal_edges
                result[u] += hubs
                result[v] += hubs
                for j in range(total):
                    x = common[j]
                    degree = indptr[x + 1] - indptr[x]
                    result[x] += (total - 1) * (base + degree - 2) + degree_sum - degree - 3 * internal_degree[x]
            else:
                valid = copies - internal_edges if induced else copies
                result[u] += valid
                result[v] += valid
                for j in range(total):
                    x = common[j]
                    result[x] += total - 1 - (internal_degree[x] if induced else 0)
            if banner:
                cycle_counts[u] += copies
                cycle_counts[v] += copies
                for j in range(total):
                    cycle_counts[common[j]] += total - 1
    if banner:
        for u in range(n):
            for p in range(indptr[u], indptr[u + 1]):
                result[u] += cycle_counts[indices[p]]
    return result


@njit
def _count_bow_ties(indptr, indices):
    """Pair incident triangles, subtracting pairs that share a second vertex."""
    n = len(indptr) - 1
    result = np.zeros(n, dtype=np.int64)
    membership = np.full(n, -1, dtype=np.int64)
    around = np.zeros(n, dtype=np.int64)
    for center in range(n):
        start, end = indptr[center], indptr[center + 1]
        if end - start < 4:
            continue
        for p in range(start, end):
            u = indices[p]
            membership[u] = center
            around[u] = 0
        triangles = np.int64(0)
        for p in range(start, end):
            u = indices[p]
            for q in range(indptr[u], indptr[u + 1]):
                v = indices[q]
                if v > u and membership[v] == center:
                    around[u] += 1
                    around[v] += 1
                    triangles += 1
        copies = _choose(triangles, 2)
        for p in range(start, end):
            copies -= _choose(around[indices[p]], 2)
        result[center] += copies
        if not copies:
            continue
        for p in range(start, end):
            u = indices[p]
            for q in range(indptr[u], indptr[u + 1]):
                v = indices[q]
                if v > u and membership[v] == center:
                    disjoint = triangles - around[u] - around[v] + 1
                    result[u] += disjoint
                    result[v] += disjoint
    return result


def count_substructure(
    edge_index: Tensor,
    num_nodes: int,
    substructure_set: set[str],
    induced: bool,
) -> dict[str, Tensor]:
    """Return per-node int64 counts of the requested three-to-five-node motifs.

    ``edge_index`` is an integer tensor with shape [2, num_edges].  Single or
    double edge directions and duplicate edges are accepted.  Node IDs must be
    in [0, num_nodes); self-loops are rejected.  Isolated nodes are retained.

    Non-induced counts distinguish copies with different edge sets, counting
    each modulo pattern automorphisms (K4 has three 4-cycles).  Induced counts
    count matching vertex sets.  Every vertex of a copy receives one count,
    regardless of its role.  The twelve accepted names are those in _PATTERNS.

    Counting runs on CPU using Numba and O(n+m) sparse storage, without a dense
    adjacency matrix.  Result tensors have shape [num_nodes] on the input
    device.  The first call includes JIT compilation.  Counts and intermediate
    combinatorial expressions must fit signed int64; no approximation is used.
    """
    if not isinstance(edge_index, Tensor):
        raise TypeError("edge_index must be a torch.Tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.layout != torch.strided:
        raise ValueError("edge_index must be a strided tensor")
    if edge_index.dtype == torch.bool or edge_index.is_floating_point() or edge_index.is_complex():
        raise TypeError("edge_index must have an integer dtype")
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral):
        raise TypeError("num_nodes must be a nonnegative integer")
    num_nodes = int(num_nodes)
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")
    if not isinstance(induced, bool):
        raise TypeError("induced must be a bool")
    if not isinstance(substructure_set, (set, frozenset)) or not all(
        isinstance(name, str) for name in substructure_set
    ):
        raise TypeError("substructure_set must be a set of pattern names")
    unknown = substructure_set.difference(_PATTERNS)
    if unknown:
        raise ValueError(f"Unknown substructures: {', '.join(sorted(unknown))}")
    edges = edge_index.detach().to(device="cpu", dtype=torch.int64).numpy()
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("edge_index contains a node outside [0, num_nodes)")
    if edges.size and np.any(edges[0] == edges[1]):
        raise ValueError("self-loops are not supported")
    if not substructure_set:
        return {}

    # Coalesce both directions using sorted pairs, avoiding overflowing n*n keys.
    directed = np.concatenate((edges, edges[::-1]), axis=1)
    if directed.shape[1]:
        order = np.lexsort((directed[1], directed[0]))
        directed = directed[:, order]
        keep = np.ones(directed.shape[1], dtype=np.bool_)
        keep[1:] = np.any(directed[:, 1:] != directed[:, :-1], axis=0)
        directed = directed[:, keep]
    indptr = np.empty(num_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(np.bincount(directed[0], minlength=num_nodes), out=indptr[1:])
    indices = np.ascontiguousarray(directed[1], dtype=np.int64)
    forward = None
    result = {}
    for name in sorted(substructure_set):
        size, pattern_edges, _ = _PATTERNS[name]
        if num_nodes < size or indices.size < 2 * len(pattern_edges):
            values = np.zeros(num_nodes, dtype=np.int64)
        elif name in ("triangle", "4-clique", "5-clique"):
            if forward is None:
                forward = _forward_csr(indptr, indices)
            values = _count_cliques(*forward, size)
        elif name in ("3-star", "4-star"):
            values = _count_stars(indptr, indices, size - 1, induced)
        elif name == "4-cycle":
            values = _count_four_cycles(indptr, indices, induced, False)
        elif name == "banner" and not induced:
            values = _count_four_cycles(indptr, indices, False, True)
        elif name == "bow-tie" and not induced:
            values = _count_bow_ties(indptr, indices)
        elif not induced and name in ("diamond", "biclique"):
            values = _count_common_neighbor_patterns(indptr, indices, name == "biclique")
        else:
            adjacency, before, degrees = _pattern_data(name)
            values = _count_embeddings(indptr, indices, adjacency, before, degrees, induced)
        result[name] = torch.from_numpy(values).to(device=edge_index.device)
    return result
