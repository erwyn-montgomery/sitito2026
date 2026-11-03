import argparse
import asyncio
from itertools import product
import json
from pathlib import Path
import sys

from ablation import K_VALUES as ABLATION_K_VALUES, PERMUTATION_SEEDS, POLICIES, load_or_create, prepare_condition
from common import writer_lock
from config import (DATASETS, GROUPINGS, K_VALUES, METRICS, MODELS, ORDERINGS, ROOT,
                    FeatureConfig, from_environment, load_environment, model_config, setting)
from dataset_io import load_scores, read_population
from inference import execute, make_plan, token_counter, validate_gold, validate_model
from prompts import PromptResources


def parser():
    p = argparse.ArgumentParser(description="Батч-классификация с учётом сложности текста")
    p.add_argument("--env-file", type=Path)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("embed", "prepare", "single", "batch", "ablate"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", choices=DATASETS, required=True)
        sub.add_argument("--csv", type=Path)
        if name in {"embed", "prepare"}:
            for flag in ("embeddings", "embedding-ids", "embedding-metadata"):
                sub.add_argument(f"--{flag}", type=Path)
            sub.add_argument("--embedding-model")
            if name == "prepare":
                sub.add_argument("--metrics", nargs="+", choices=[*METRICS, "all"], default=["all"])
                for flag in ("output-csv", "output-metadata", "label-cache"):
                    sub.add_argument(f"--{flag}", type=Path)
                sub.add_argument("--trusted-pickle-ids", action="store_true")
                sub.add_argument("--count-tokens", action="store_true",
                                 help="Вычислить token_len документов выбранным токенизатором")
                sub.add_argument("--length-tokenizer")
        else:
            sub.add_argument("--metadata", type=Path)
            sub.add_argument("--model", nargs="+", choices=[*MODELS, "all"], default=["all"])
            sub.add_argument("--few-shot", type=Path)
            sub.add_argument("--output-root", type=Path)
            if name != "ablate":
                sub.add_argument("--seed", type=int, default=42)
            if name == "batch":
                sub.add_argument("--grouping", choices=[*GROUPINGS, "all"], default="all")
                sub.add_argument("--ordering", choices=[*ORDERINGS, "all"], default="all")
                sub.add_argument("--difficulty", nargs="+", choices=[*METRICS, "all"], default=["all"])
                sub.add_argument("--k", nargs="+", type=int, choices=K_VALUES, default=list(K_VALUES))
            if name == "ablate":
                sub.add_argument("--difficulty", nargs="+", choices=[*METRICS, "all"], default=["all"])
                sub.add_argument("--policy", nargs="+", choices=POLICIES, default=list(POLICIES))
                sub.add_argument("--k", nargs="+", type=int, choices=ABLATION_K_VALUES, default=list(ABLATION_K_VALUES))
                sub.add_argument("--permutation-seeds", nargs="+", type=int, default=list(PERMUTATION_SEEDS))
                sub.add_argument("--execute", action="store_true")
    return p


def unique(values):
    return list(dict.fromkeys(values))


async def run(args, paths):
    population = read_population(args.dataset, paths["csv"])
    prompts = PromptResources.load(ROOT / "prompts" / f"{args.dataset}.txt", paths["few_shot"], population.ids)
    models = list(MODELS) if "all" in args.model else unique(args.model)
    actual_inference = args.command != "ablate" or args.execute
    if actual_inference:
        validate_gold(population)
    metrics = list(METRICS) if "all" in getattr(args, "difficulty", []) else unique(getattr(args, "difficulty", []))
    scores = {}

    def provenance(metric):
        if metric not in scores:
            scores[metric] = load_scores(population, metric, paths["metadata"])
        return scores[metric][1]

    dependent = args.command == "ablate" or (args.command == "batch" and (args.grouping, args.ordering) != ("bfd", "random"))
    if dependent:
        for metric in metrics:
            provenance(metric)
    configurations = [model_config(name, execute=actual_inference) for name in models]
    for model in configurations:
        validate_model(model)
    for model in configurations:
        model_name = model.identity
        counter = token_counter(model)
        if args.command == "single":
            jobs = [("single_doc", "identity", None, 1)]
        elif args.command == "batch":
            groupings = GROUPINGS if args.grouping == "all" else [args.grouping]
            orderings = ORDERINGS if args.ordering == "all" else [args.ordering]
            jobs = [(g, o, m, k) for g, o in product(groupings, orderings)
                    for m in ([None] if (g, o) == ("bfd", "random") else metrics)
                    for k in unique(args.k)]
        else:
            for metric, permutation_seed in product(metrics, unique(args.permutation_seeds)):
                detail = provenance(metric)
                raw = dict(zip(population.ids, scores[metric][0].tolist()))
                path = paths["output_root"] / "permutations" / args.dataset / metric / f"seed_{permutation_seed}.json"
                manifest = load_or_create(path, args.dataset, metric, permutation_seed,
                                           population.ids, population.lengths, raw)
                for policy, k in product(unique(args.policy), unique(args.k)):
                    grouping, ordering = policy.split("/")
                    plan, directory = prepare_condition(population, model, prompts, paths["output_root"],
                        grouping=grouping, ordering=ordering, metric=metric, k=k,
                        manifest=manifest, provenance=detail, counter=counter)
                    if args.execute:
                        print(json.dumps(await execute(plan, population, model, directory), ensure_ascii=False))
                    else:
                        print(json.dumps(dict(plan=str(directory / "plan.json"), inference=False), ensure_ascii=False))
            continue
        for grouping, ordering, metric, k in jobs:
            detail = provenance(metric) if metric else None
            plan = make_plan(population, model, prompts, grouping=grouping, ordering=ordering,
                             metric=metric, k=k, seed=args.seed, provenance=detail, counter=counter)
            directory = (paths["output_root"] / model_name / args.dataset / f"{grouping}__{ordering}"
                         / (metric or "control") / f"seed_{args.seed}" / f"k_{k}")
            print(json.dumps(await execute(plan, population, model, directory), ensure_ascii=False))


def describe_error(exc):
    children = getattr(exc, "exceptions", None)
    return "; ".join(describe_error(child) for child in children) if children else str(exc)


def main(argv=None):
    args = parser().parse_args(argv)
    load_environment(args.env_file)
    try:
        if args.command in {"embed", "prepare"}:
            from difficulty import embed, prepare
            names = ("csv", "output_csv", "output_metadata", "embeddings", "embedding_ids", "embedding_metadata", "label_cache")
            required = {"csv", "embeddings", "embedding_ids"} if args.command == "embed" else {"csv", "output_csv"}
            paths = {name: setting(args, name, required=name in required) for name in names}
            overrides = {"embedding_model": args.embedding_model} if args.embedding_model is not None else {}
            config = from_environment(FeatureConfig, "FEATURES", **overrides)
            output = paths["embeddings"] if args.command == "embed" else paths["output_csv"]
            with writer_lock(output.parent):
                result = embed(args, config, paths) if args.command == "embed" else prepare(args, config, paths)
                print(json.dumps(result, ensure_ascii=False))
        else:
            paths = {name: setting(args, name, required=name in {"csv", "output_root"})
                     for name in ("csv", "metadata", "few_shot", "output_root")}
            with writer_lock(paths["output_root"]):
                asyncio.run(run(args, paths))
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Запуск не завершён: {describe_error(exc) or 'прерывание'}. Предыдущие успешные результаты сохранены.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
