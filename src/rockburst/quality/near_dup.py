"""Tolerance-based near-duplicate matching on raw measured inputs."""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Raw measured inputs only (not derived SCF / B1 / B2).
RAW_DUP_COLS: tuple[str, ...] = (
    "sigma_theta",
    "sigma_c",
    "sigma_t",
    "wet",
    "d",
)

DEFAULT_REL_TOL = 0.01
MIN_OVERLAP = 3


def rows_near_duplicate(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    min_overlap: int = MIN_OVERLAP,
) -> bool:
    """True if every overlapping finite feature differs by < rel_tol relatively.

    Relative difference: ``|a-b| / max(|a|, |b|, eps) < rel_tol``.
    Features that are NaN on either side are ignored (\"when present\").
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    both = np.isfinite(a) & np.isfinite(b)
    if int(both.sum()) < min_overlap:
        return False
    aa, bb = a[both], b[both]
    denom = np.maximum(np.maximum(np.abs(aa), np.abs(bb)), 1e-12)
    return bool(np.all(np.abs(aa - bb) / denom < rel_tol))


def pairwise_near_dup_mask(
    feats: np.ndarray,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    min_overlap: int = MIN_OVERLAP,
) -> np.ndarray:
    """Symmetric boolean adjacency (n×n) for near-duplicate pairs."""
    n = feats.shape[0]
    adj = np.eye(n, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if rows_near_duplicate(
                feats[i], feats[j], rel_tol=rel_tol, min_overlap=min_overlap
            ):
                adj[i, j] = adj[j, i] = True
    return adj


def connected_components(adj: np.ndarray) -> list[list[int]]:
    """Union-find connected components; each component sorted by index."""
    n = adj.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Prefer smaller root for determinism
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                union(i, j)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        comps.setdefault(r, []).append(i)
    return [sorted(v) for _, v in sorted(comps.items(), key=lambda kv: min(kv[1]))]


def assign_component_ids(
    components: Sequence[Sequence[int]],
    *,
    order_keys: np.ndarray | None = None,
) -> np.ndarray:
    """Map row index → group id. Components ordered by min order_key (e.g. ``no``)."""
    n = sum(len(c) for c in components)
    ids = np.full(n, -1, dtype=int)
    if order_keys is None:
        ordered = sorted(components, key=lambda c: min(c))
    else:
        ordered = sorted(components, key=lambda c: int(order_keys[min(c)]))
    for gid, comp in enumerate(ordered):
        for i in comp:
            ids[i] = gid
    return ids
