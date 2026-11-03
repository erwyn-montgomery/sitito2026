from collections import Counter
from copy import copy
import math
import random

from common import fingerprint, read_json, save_json
from inference import make_plan

VERSION = "length-stratified-raw-permutation-v1"
POLICIES = ("diff_diverse/random", "bfd/diff_edges")
K_VALUES = (8, 16, 32, 64)
PERMUTATION_SEEDS = (101, 202, 303)


def length_strata(ids, lengths):
    if (len(ids) != len(set(ids)) or not ids
            or any(not isinstance(i, str) or not i or i.strip() != i for i in ids)):
        raise ValueError("Permutation IDs must be unique nonempty strings")
    if set(lengths) != set(ids) or any(type(lengths[i]) is not int or lengths[i] <= 0 for i in ids):
        raise ValueError("Aligned positive integer canonical lengths are required")
    ordered = sorted(ids, key=lambda i: (lengths[i], i))
    quotient, remainder = divmod(len(ordered), 10)
    strata, start = [], 0
    for index in range(10):
        end = start + quotient + (index < remainder)
        strata.append(ordered[start:end])
        start = end
    return strata


def assignment(dataset, metric, seed, ids, lengths, raw_scores):
    ids = tuple(ids)
    if set(raw_scores) != set(ids) or any(not math.isfinite(v) for v in raw_scores.values()):
        raise ValueError("Finite difficulty scores must align exactly with IDs")
    strata = length_strata(ids, lengths)
    rng_seed = int(fingerprint([VERSION, dataset, metric, int(seed)]), 16)
    rng = random.Random(rng_seed)
    donors, stratum_for = {}, {}
    for index, stratum in enumerate(strata):
        shuffled = list(stratum)
        rng.shuffle(shuffled)
        donors.update(zip(stratum, shuffled))
        stratum_for.update((identifier, index) for identifier in stratum)
    rows = [{"document_id": i, "donor_id": donors[i], "stratum": stratum_for[i],
             "token_len": lengths[i], "original_raw_score": raw_scores[i],
             "assigned_raw_score": raw_scores[donors[i]]} for i in ids]
    manifest = dict(version=VERSION, dataset=dataset, metric=metric, permutation_seed=int(seed),
        rng={"implementation": "python.random.Random/MT19937", "derived_seed": str(rng_seed)},
        ordered_ids_sha256=fingerprint(ids),
        canonical_lengths_sha256=fingerprint([(i, lengths[i]) for i in ids]),
        original_assignment_sha256=fingerprint([(i, raw_scores[i]) for i in ids]),
        score_assignment_sha256=fingerprint([(r["document_id"], r["assigned_raw_score"]) for r in rows]),
        strata=strata, strata_sha256=fingerprint(strata), assignments=rows,
        effective_change_fraction=sum(r["original_raw_score"] != r["assigned_raw_score"] for r in rows) / len(ids))
    manifest["manifest_sha256"] = fingerprint(manifest)
    validate_assignment(manifest, dataset, metric, seed, ids, lengths, raw_scores)
    return manifest


def validate_assignment(manifest, dataset, metric, seed, ids, lengths, raw_scores):
    contents = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != fingerprint(contents):
        raise ValueError("Permutation manifest checksum mismatch")
    strata = length_strata(ids, lengths)
    checks = dict(version=VERSION, dataset=dataset, metric=metric, permutation_seed=seed,
        ordered_ids_sha256=fingerprint(ids), strata=strata, strata_sha256=fingerprint(strata),
        canonical_lengths_sha256=fingerprint([(i, lengths[i]) for i in ids]),
        original_assignment_sha256=fingerprint([(i, raw_scores[i]) for i in ids]))
    if any(manifest.get(key) != value for key, value in checks.items()):
        raise ValueError("Permutation inputs changed")
    rows = manifest["assignments"]
    if [r["document_id"] for r in rows] != list(ids):
        raise ValueError("Permutation document alignment mismatch")
    by_id = {r["document_id"]: r for r in rows}
    for index, stratum in enumerate(strata):
        if Counter(by_id[i]["donor_id"] for i in stratum) != Counter(stratum):
            raise ValueError("Donors must form a within-stratum permutation")
        for i in stratum:
            row = by_id[i]
            if (row["stratum"] != index or row.get("token_len") != lengths[i]
                    or row["original_raw_score"] != raw_scores[i]
                    or row["assigned_raw_score"] != raw_scores[row["donor_id"]]):
                raise ValueError("Invalid reassigned score")
        if Counter(raw_scores[i] for i in stratum) != Counter(by_id[i]["assigned_raw_score"] for i in stratum):
            raise ValueError("Within-stratum raw-score multiset changed")
    if Counter(raw_scores.values()) != Counter(r["assigned_raw_score"] for r in rows):
        raise ValueError("Overall raw-score multiset changed")
    if manifest["score_assignment_sha256"] != fingerprint([(r["document_id"], r["assigned_raw_score"]) for r in rows]):
        raise ValueError("Score assignment checksum mismatch")
    if manifest["effective_change_fraction"] != sum(r["original_raw_score"] != r["assigned_raw_score"] for r in rows) / len(ids):
        raise ValueError("Incorrect effective change fraction")


def load_or_create(path, dataset, metric, seed, ids, lengths, raw_scores):
    if path.exists():
        manifest = read_json(path)
        validate_assignment(manifest, dataset, metric, seed, ids, lengths, raw_scores)
    else:
        manifest = assignment(dataset, metric, seed, ids, lengths, raw_scores)
        save_json(path, manifest)
    return manifest


def freeze_json(path, value):
    if path.exists():
        if fingerprint(read_json(path)) != fingerprint(value):
            raise ValueError(f"Сохранённый план изменился: {path}; выберите другой output-root")
    else:
        save_json(path, value)


def plan_diagnostics(original, permuted, lengths):
    original_groups = [b["grouped_ids"] for b in original["batches"]]
    permuted_groups = [b["grouped_ids"] for b in permuted["batches"]]
    before = [b["document_ids"] for b in original["batches"]]
    after = [b["document_ids"] for b in permuted["batches"]]
    canonical = lambda groups: sorted(sorted(group) for group in groups)
    membership_equal = canonical(original_groups) == canonical(permuted_groups)
    if original["condition"]["grouping"] == "bfd" and not membership_equal:
        raise ValueError("BFD membership changed under score reassignment")
    before_members = {i: frozenset(group) for group in original_groups for i in group}
    after_members = {i: frozenset(group) for group in permuted_groups for i in group}
    before_positions = {i: [batch, position] for batch, group in enumerate(before) for position, i in enumerate(group)}
    after_positions = {i: [batch, position] for batch, group in enumerate(after) for position, i in enumerate(group)}
    requests = lambda plan: [b["request_sha256"] for b in plan["batches"]]
    requests_equal = requests(original) == requests(permuted)
    changed_members = [i for i in lengths if before_members[i] != after_members[i]]
    changed_positions = [i for i in lengths if before_positions[i] != after_positions[i]]
    before_requests, after_requests = requests(original), requests(permuted)
    changed_requests = [index for index in range(max(len(before_requests), len(after_requests)))
                        if index >= len(before_requests) or index >= len(after_requests)
                        or before_requests[index] != after_requests[index]]
    return dict(membership_identical=membership_equal, ordered_batches_identical=before == after,
        rendered_requests_identical=requests_equal, operationally_inert=requests_equal,
        membership_changed_document_ids=changed_members, position_changed_document_ids=changed_positions,
        membership_changed_document_fraction=len(changed_members) / len(lengths),
        position_changed_document_fraction=len(changed_positions) / len(lengths),
        changed_cobatched_pair_count=sum(len(before_members[i] ^ after_members[i]) for i in lengths) // 2,
        positions_before=before_positions, positions_after=after_positions,
        changed_request_indices=changed_requests, original_request_hashes=before_requests,
        permuted_request_hashes=after_requests,
        original_membership_hash=fingerprint(canonical(original_groups)),
        permuted_membership_hash=fingerprint(canonical(permuted_groups)),
        original_order_hash=fingerprint(before), permuted_order_hash=fingerprint(after),
        reference_authority="computed_reference_plan")


def prepare_condition(population, model, prompts, root, *, grouping, ordering, metric, k,
                      manifest, provenance, counter):
    kwargs = dict(grouping=grouping, ordering=ordering, metric=metric, k=k, seed=42,
                  provenance=provenance, counter=counter)
    original = make_plan(population, model, prompts, **kwargs)
    original_path = root / "reference_plans" / model.identity / population.dataset / f"{grouping}__{ordering}" / metric / f"k_{k}.json"
    freeze_json(original_path, original)
    intervened = copy(population)
    intervened.frame = population.frame.copy(deep=True)
    assigned = {r["document_id"]: r["assigned_raw_score"] for r in manifest["assignments"]}
    column = f"diff_{metric}_raw"
    intervened.frame[column] = [assigned[i] for i in population.ids]
    if not intervened.frame.drop(columns=[column]).equals(population.frame.drop(columns=[column])):
        raise ValueError("Intervention modified non-score document attributes")
    plan = make_plan(intervened, model, prompts, **kwargs)
    intervention = dict(kind="length_stratified_raw_permutation", version=VERSION,
                        permutation_seed=manifest["permutation_seed"],
                        manifest_sha256=manifest["manifest_sha256"],
                        score_assignment_sha256=manifest["score_assignment_sha256"],
                        original_assignment_sha256=manifest["original_assignment_sha256"],
                        strata_sha256=manifest["strata_sha256"])
    plan["condition"]["score_intervention"] = intervention
    plan["condition_id"] = fingerprint(plan["condition"])
    plan["experiment_id"] = fingerprint([plan["condition_id"], k])
    directory = (root / model.identity / population.dataset / f"{grouping}__{ordering}" / metric
                 / f"permutation_{manifest['permutation_seed']}" / f"k_{k}")
    diagnostic = dict(plan_diagnostics(original, plan, population.lengths),
                      effective_change_fraction=manifest["effective_change_fraction"],
                      manifest_sha256=manifest["manifest_sha256"], condition_id=plan["condition_id"], k=k)
    freeze_json(directory / "plan.json", plan)
    freeze_json(directory / "diagnostics.json", diagnostic)
    return plan, directory
