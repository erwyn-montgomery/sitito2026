from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import pandas as pd

from common import stable_seed
from config import ALGORITHMS


@dataclass(frozen=True)
class Item:
    id: str
    length: int
    payload: Any
    score: float = 0.0


def items(df, *, k, l, id_col="index", len_col="token_len", text_col="text",
          payload_cols=None, diff_col=None, return_singleton_as_str=True):
    if type(k) is not int or k <= 0 or type(l) is not int or l <= 0:
        raise ValueError("k and l must be positive integers")
    ids = df[id_col].astype(str).tolist()
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Missing/duplicate string document IDs")
    lengths = pd.to_numeric(df[len_col], errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(lengths).all() or (lengths <= 0).any()
            or (lengths != np.floor(lengths)).any()):
        raise ValueError("Invalid canonical lengths")
    if (lengths > l).any():
        raise ValueError("INFEASIBLE_CONTENT: document exceeds fixed content budget")
    scores = np.zeros(len(df))
    if diff_col:
        scores = pd.to_numeric(df[diff_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("Algorithms require finite difficulty scores")
    spec = payload_cols if payload_cols is not None else text_col
    result = []
    for i, row in enumerate(df.to_dict("records")):
        if isinstance(spec, str):
            payload = str(row[spec]) if return_singleton_as_str else {spec: row[spec]}
        elif isinstance(spec, dict):
            payload = {out: row[source] for out, source in spec.items()}
        else:
            payload = {name: row[name] for name in spec}
        result.append(Item(ids[i], int(lengths[i]), payload, float(scores[i])))
    return result


def pack(group, k, l, span=None):
    bins = []
    for item in sorted(group, key=lambda x: (-x.length, x.score, x.id)):
        candidates = []
        for i, bucket in enumerate(bins):
            total = sum(x.length for x in bucket) + item.length
            scores = [x.score for x in bucket] + [item.score]
            if len(bucket) < k and total <= l and (span is None or max(scores) - min(scores) <= span + 1e-12):
                candidates.append((l - total, k - len(bucket) - 1, i))
        if candidates:
            bins[min(candidates)[2]].append(item)
        else:
            bins.append([item])
    return bins


def materialize(bins):
    return [{item.id: item.payload for item in bucket} for bucket in bins]


def validate_partition(ids, batches, k, lengths, l):
    flattened = [i for batch in batches for i in batch]
    if (len(flattened) != len(ids) or len(flattened) != len(set(flattened))
            or set(flattened) != set(ids)):
        raise ValueError("Batching must preserve every document exactly once")
    for batch in batches:
        if not batch or len(batch) > k or sum(lengths[i] for i in batch) > l:
            raise ValueError("Batch count/content constraint violated")


def validate_permutation(before, after):
    if len(after) != len(before) or len(set(after)) != len(after) or set(after) != set(before):
        raise ValueError("Orderer must return an exact document ID permutation")


def kmeans_1d(x, clusters, rng, *, n_init=10, max_iter=100, tol=1e-4):
    x = np.asarray(x, dtype=float)
    best = None
    for _ in range(n_init):
        centers = [float(x[rng.integers(len(x))])]
        while len(centers) < clusters:
            distances = ((x[:, None] - np.array(centers)[None, :]) ** 2).min(axis=1)
            index = rng.choice(len(x), p=distances / distances.sum()) if distances.sum() > 0 else rng.integers(len(x))
            centers.append(float(x[index]))
        centers = np.array(centers)
        for _ in range(max_iter):
            labels = ((x[:, None] - centers[None, :]) ** 2).argmin(axis=1)
            updated = np.array([x[labels == j].mean() if (labels == j).any()
                                else x[rng.integers(len(x))] for j in range(clusters)])
            done = np.max(np.abs(updated - centers)) <= tol
            centers = updated
            if done:
                break
        labels = ((x[:, None] - centers[None, :]) ** 2).argmin(axis=1)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (centers.copy(), labels.copy(), inertia)
    return best


def make_batches_bfd(df, k, l, **kwargs):
    kwargs.pop("diff_col", None)
    bundle = items(df, k=k, l=l, **kwargs)
    return materialize(pack(bundle, k, l))


def make_batches_diff_close(df, k, l, *, diff_col, seed=42, config=ALGORITHMS, **kwargs):
    group = items(df, k=k, l=l, diff_col=diff_col, **kwargs)
    if not group:
        return []
    scores = np.array([x.score for x in group])
    maximum = min(len(group), len(np.unique(scores)), min(30, max(5, round(np.sqrt(len(group))))))
    rng = np.random.default_rng(seed)
    fits = [kmeans_1d(scores, c, rng, n_init=config.kmeans_n_init,
                      max_iter=config.kmeans_max_iter, tol=config.kmeans_tol)
            for c in range(1, maximum + 1)]
    inertia = np.array([fit[2] for fit in fits])
    chosen = 0
    if maximum > 1 and inertia.max() > inertia.min():
        gaps = 1 - (inertia - inertia.min()) / (inertia.max() - inertia.min()) - np.linspace(0, 1, maximum)
        if gaps.max() >= config.knee_min_strength:
            chosen = int(gaps.argmax())
    _, labels, _ = kmeans_1d(scores, chosen + 1, rng, n_init=config.kmeans_n_init,
                            max_iter=config.kmeans_max_iter, tol=config.kmeans_tol)
    subgroups = []
    for label in sorted(set(labels.tolist())):
        ordered = sorted([item for item, code in zip(group, labels) if code == label],
                         key=lambda x: (x.score, x.id))
        current = []
        for item in ordered:
            if current and item.score - current[0].score > config.cluster_span + 1e-12:
                subgroups.append(current)
                current = []
            current.append(item)
        if current:
            subgroups.append(current)
    bins = []
    for subgroup in sorted(subgroups, key=lambda g: (-len(g), np.median([x.score for x in g]))):
        bins.extend(pack(subgroup, k, l, span=config.batch_span))
    bins.sort(key=lambda b: (-len(b), np.median([x.score for x in b])))
    merged = []
    while bins:
        base = bins.pop(0)
        while True:
            candidates = []
            for j, other in enumerate(bins):
                combined = base + other
                span = max(x.score for x in combined) - min(x.score for x in combined)
                total = sum(x.length for x in combined)
                if len(combined) <= k and total <= l and span <= config.batch_span + 1e-12:
                    candidates.append((span, abs(np.median([x.score for x in base]) - np.median([x.score for x in other])), l - total, j))
            if not candidates:
                break
            base += bins.pop(min(candidates)[-1])
        merged.append(sorted(base, key=lambda x: (x.score, x.id)))
    merged.sort(key=lambda b: (np.median([x.score for x in b]), sum(x.length for x in b), len(b)))
    return materialize(merged)


def make_batches_diff_diverse(df, k, l, *, diff_col, seed=42, **kwargs):
    group = items(df, k=k, l=l, diff_col=diff_col, **kwargs)
    if len(group) < 2 or k == 1:
        return materialize(pack(group, k, l))
    count = min(len(group), 32, max(2, min(k, round(np.sqrt(len(group))))))
    codes = np.asarray(pd.qcut([x.score for x in group], count, labels=False, duplicates="drop"))
    codes = np.nan_to_num(codes, nan=0).astype(int)
    strata = {int(code): [] for code in sorted(set(codes.tolist()))}
    for item, code in zip(group, codes):
        strata[int(code)].append(item)
    if len(strata) <= 1:
        return materialize(pack(group, k, l))
    for bucket in strata.values():
        bucket.sort(key=lambda x: (x.length, x.id))
    keys, zigzag = sorted(strata), []
    while keys:
        zigzag.append(keys.pop(0))
        if keys:
            zigzag.append(keys.pop())

    def pop_fit(code, remaining):
        bucket = strata[code]
        for j in range(len(bucket) - 1, -1, -1):
            if bucket[j].length <= remaining:
                return bucket.pop(j)
        return None

    bins = []
    while any(strata.values()):
        batch, used = [], 0
        for code in (min(strata), max(strata)):
            item = pop_fit(code, l - used)
            if item is not None and len(batch) < k:
                batch.append(item)
                used += item.length
        position = len(bins) % len(zigzag)
        while len(batch) < k:
            added = False
            for _ in zigzag:
                code = zigzag[position % len(zigzag)]
                position += 1
                item = pop_fit(code, l - used)
                if item is not None:
                    batch.append(item)
                    used += item.length
                    added = True
                    break
            if not added:
                break
        if not batch:
            raise RuntimeError("Validated nonoversized documents could not be packed")
        bins.append(batch)
    return materialize(bins)


def make_batches_diff_routing(df, k, l, *, diff_col, seed=42, config=ALGORITHMS, **kwargs):
    group = items(df, k=k, l=l, diff_col=diff_col, **kwargs)
    if k == 1 or config.routing_strength <= 0:
        return materialize(pack(group, k, l))
    bins = []

    def utilization(bucket):
        return max(sum(x.length for x in bucket) / l, len(bucket) / k)

    for item in sorted(group, key=lambda x: (-x.score, -x.length, x.id)):
        cap = min(1., max(0.05, 1 - min(.95, config.routing_strength) * item.score))
        chosen = None
        for relaxation in range(config.max_relax_iters + 1):
            candidates = []
            limit = min(1., cap + relaxation * config.relax_step)
            for j, bucket in enumerate(bins):
                if len(bucket) < k and sum(x.length for x in bucket) + item.length <= l:
                    u = utilization(bucket + [item])
                    if u <= limit + 1e-12:
                        candidates.append((u, sum(x.length for x in bucket), j))
            if candidates:
                chosen = min(candidates)[-1]
                break
        if chosen is None:
            bins.append([item])
        else:
            bins[chosen].append(item)
    indices = sorted(range(len(bins)), key=lambda j: (max(x.score for x in bins[j]), utilization(bins[j])))
    for j in indices:
        bucket = bins[j]
        if not bucket or max(x.score for x in bucket) > config.easy_hardness_threshold:
            continue
        for item in sorted(bucket.copy(), key=lambda x: (x.score, x.length, x.id)):
            destinations = sorted([i for i, b in enumerate(bins) if i != j and b
                                   and max(x.score for x in b) <= config.compact_into_max_hardness],
                                  key=lambda i: (utilization(bins[i]), sum(x.length for x in bins[i])), reverse=True)
            destination = next((i for i in destinations if len(bins[i]) < k
                                and sum(x.length for x in bins[i]) + item.length <= l), None)
            if destination is None:
                break
            bins[destination].append(item)
            bucket.remove(item)
    bins = [b for b in bins if b]
    bins.sort(key=lambda b: (-max(x.score for x in b), utilization(b), sum(x.length for x in b), len(b)))
    return materialize(bins)


def order_random(doc_ids, *, seed=42, **_):
    result = sorted(doc_ids)
    random.Random(stable_seed(seed, result)).shuffle(result)
    return result


def order_diff_edges(doc_ids, *, scores, lengths, **_):
    ordered = sorted(doc_ids, key=lambda i: (-scores[i], -lengths[i], i))
    result = [None] * len(ordered)
    left, right = 0, len(result) - 1
    for j, identifier in enumerate(ordered):
        if j % 2 == 0:
            result[left] = identifier
            left += 1
        else:
            result[right] = identifier
            right -= 1
    return result


def order_diff_disperse(doc_ids, *, scores, lengths, **_):
    ordered = sorted(doc_ids, key=lambda i: (scores[i], -lengths[i], i))
    result = []
    low, high = 0, len(ordered) - 1
    while low <= high:
        result.append(ordered[low])
        low += 1
        if low <= high:
            result.append(ordered[high])
            high -= 1
    return result


GROUPERS = {"bfd": make_batches_bfd, "diff_close": make_batches_diff_close,
            "diff_diverse": make_batches_diff_diverse, "diff_routing": make_batches_diff_routing}
ORDERERS = {"random": order_random, "diff_edges": order_diff_edges, "diff_disperse": order_diff_disperse}
