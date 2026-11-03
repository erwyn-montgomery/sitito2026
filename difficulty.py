from dataclasses import asdict
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, softmax, xlogy

from common import atomic_text, code_hashes, file_hash, fingerprint, read_json, save_json, save_npy, utc_now, versions
from config import DATASETS, FEATURE_FORMAT, METRICS, embedding_settings
from dataset_io import label_resource, load_embeddings, read_corpus, read_population, validate_scores


def bounded(values, *, tolerance=1e-9):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < -tolerance).any() or (values > 1 + tolerance).any():
        raise ValueError("Material violation of the theoretical [0,1] score bound")
    return np.clip(values, 0., 1.)


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid probability vector")
    if not np.allclose(p.sum(axis=-1), 1., atol=1e-9):
        raise ValueError("Probability distributions must sum to one")
    return -xlogy(p, p).sum(axis=-1)


def readability(texts):
    import textstat
    textstat.set_lang("en_US")
    scores = np.array([textstat.flesch_kincaid_grade(text) for text in texts], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite readability output")
    return scores, dict(method="textstat.flesch_kincaid_grade", language="en_US")


def score_token_ids(ids, model, *, context_tokens, stride, device):
    import torch
    if len(ids) < 2 or not 1 <= stride < context_tokens:
        raise ValueError("PPL requires >=2 tokens and 1 <= stride < context size")
    total_nll, scored, forwarded, forwards = 0., 0, 0, 0
    target_start = 1
    with torch.inference_mode():
        while target_start < len(ids):
            end = min(len(ids), target_start + stride)
            start = max(0, end - context_tokens)
            tensor = torch.tensor([ids[start:end]], device=device)
            logits = model(input_ids=tensor, use_cache=False).logits.float()
            first = target_start - start
            targets = tensor[:, first:]
            loss = torch.nn.functional.cross_entropy(
                logits[:, first - 1:-1, :].reshape(-1, logits.shape[-1]),
                targets.reshape(-1), reduction="sum")
            total_nll += float(loss.item())
            scored += end - target_start
            forwarded += end - start
            forwards += 1
            target_start = end
    if scored != len(ids) - 1 or not math.isfinite(total_nll):
        raise ValueError("PPL next-token completeness/numerical check failed")
    average = total_nll / scored
    value = math.exp(average)
    if not math.isfinite(value):
        raise ValueError("Nonfinite PPL")
    return value, dict(avg_nll=average, scored_tokens=scored,
                       auxiliary_input_tokens=forwarded, forward_passes=forwards)


def perplexity(texts, config):
    import gc
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not 1 <= config.ppl_stride < config.ppl_context_tokens:
        raise ValueError("PPL requires 1 <= stride < context size")
    tokenizer = AutoTokenizer.from_pretrained(config.ppl_model)
    model = AutoModelForCausalLM.from_pretrained(config.ppl_model,
                                               torch_dtype=getattr(torch, config.torch_dtype))
    model.to(config.device).eval()
    if config.ppl_context_tokens > model.config.max_position_embeddings:
        raise ValueError("Configured PPL window exceeds checkpoint position limit")
    values, records = [], []
    try:
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False, truncation=False).input_ids
            value, counts = score_token_ids(ids, model, context_tokens=config.ppl_context_tokens,
                                            stride=config.ppl_stride, device=config.device)
            values.append(value)
            records.append(counts)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.array(values), dict(model=config.ppl_model,
        rendering="document_only_no_chat_no_BOS", long_context="sliding_windows_each_target_once",
        context_tokens=config.ppl_context_tokens, stride=config.ppl_stride, dtype=config.torch_dtype,
        per_document=records, auxiliary_input_tokens=sum(r["auxiliary_input_tokens"] for r in records),
        forward_passes=sum(r["forward_passes"] for r in records))


def knn_distance(vectors, *, neighbors=20, query_batch_size=256):
    import faiss
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or len(x) < 2 or neighbors < 1 or query_batch_size < 1:
        raise ValueError("kNN requires at least two documents and positive search parameters")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or (norms <= 1e-12).any():
        raise ValueError("Nonfinite/zero embedding")
    x = np.ascontiguousarray(x / norms)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    count = min(neighbors, len(x) - 1)
    result = np.empty(len(x), dtype=float)
    for start in range(0, len(x), query_batch_size):
        similarity, indices = index.search(x[start:start + query_batch_size], count + 1)
        for local, (sims, rows) in enumerate(zip(similarity, indices)):
            allowed = [(int(row), float(sim)) for row, sim in zip(rows, sims)
                       if row >= 0 and row != start + local][:count]
            if len(allowed) != count:
                raise ValueError("Incomplete nonself neighbor search")
            result[start + local] = np.mean([1 - np.clip(sim, -1, 1) for _, sim in allowed])
    return result, dict(method="FAISS_IndexFlatIP_exact_cosine", neighbors_requested=neighbors,
                        neighbors_effective=count, self_exclusion="row_identity",
                        reference="entire_unlabeled_population", query_batch_size=query_batch_size)


def encode_embeddings(texts, config):
    if (not config.embedding_model or config.embedding_max_seq_length < 1 or config.embedding_batch_size < 1
            or config.embedding_padding_side not in ("left", "right")):
        raise ValueError("Задайте модель и корректные параметры кодирования эмбеддингов")
    import gc
    import torch
    from sentence_transformers import SentenceTransformer
    texts = list(texts)
    model = SentenceTransformer(config.embedding_model,
        device=config.device, model_kwargs={"torch_dtype": getattr(torch, config.torch_dtype)})
    model.max_seq_length = int(config.embedding_max_seq_length)
    model.tokenizer.padding_side = config.embedding_padding_side
    try:
        vectors = np.asarray(model.encode(texts, batch_size=config.embedding_batch_size,
            normalize_embeddings=True, convert_to_numpy=True, prompt=""), dtype=np.float32)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] < 1:
        raise ValueError("Неверная размерность эмбеддингов")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError("Некорректные или нулевые эмбеддинги")
    return vectors


def embed(args, config, paths):
    output, ids_path = paths["embeddings"], paths["embedding_ids"]
    metadata_path = paths["embedding_metadata"] or output.with_suffix(".metadata.json")
    if output.suffix != ".npy" or ids_path.suffix != ".json":
        raise ValueError("Для эмбеддингов укажите .npy, для ID — .json")
    if len({paths["csv"], output, ids_path, metadata_path}) != 4:
        raise ValueError("Входной CSV и три выходных файла должны иметь разные пути")
    corpus = read_corpus(args.dataset, paths["csv"])
    vectors = encode_embeddings(corpus.texts, config)
    save_npy(output, vectors)
    save_json(ids_path, list(corpus.ids))
    metadata = dict(format="document-embeddings-v1", dataset=args.dataset,
        document_ids=list(corpus.ids), shape=list(vectors.shape),
        score_text_fields=list(DATASETS[args.dataset]["scoring"]),
        scoring_content_by_id_sha256=fingerprint(sorted(zip(corpus.ids, corpus.texts))),
        source_csv_sha256=file_hash(paths["csv"]), embeddings_sha256=file_hash(output),
        ids_file_sha256=file_hash(ids_path), embedding_config=embedding_settings(config),
        device=config.device, normalize_embeddings=True, prompt="",
        long_text_policy="encoder_max_seq_length", computed_at=utc_now(),
        software=versions(), code=code_hashes())
    save_json(metadata_path, metadata)
    return dict(embeddings=str(output), embedding_ids=str(ids_path), metadata=str(metadata_path),
                documents=len(corpus.ids), dimension=vectors.shape[1])


def label_embeddings(resource, config, cache_dir):
    if (config.embedding_max_seq_length < 1 or config.embedding_batch_size < 1
            or config.embedding_padding_side not in ("left", "right")):
        raise ValueError("Invalid label embedding settings")
    identity = dict(hypotheses=resource["positive"], encoder=config.embedding_model,
                    dtype=config.torch_dtype,
                    max_seq_length=config.embedding_max_seq_length,
                    tokenizer_padding_side=config.embedding_padding_side,
                    encoding_mode="symmetric_no_instruction")
    path = Path(cache_dir) / f"labels_{fingerprint(identity)}.json"
    if path.is_file():
        saved = read_json(path)
        if saved["identity"] != identity or saved["vectors_sha256"] != fingerprint(saved["vectors"]):
            raise ValueError("Corrupt label-embedding cache")
        vectors = np.array(saved["vectors"], dtype=np.float32)
    else:
        vectors = encode_embeddings(resource["positive"], config)
        save_json(path, dict(identity=identity, vectors=vectors.tolist(),
                             vectors_sha256=fingerprint(vectors.tolist())))
    if (vectors.ndim != 2 or vectors.shape[0] != len(resource["labels"])
            or vectors.shape[1] < 1 or not np.isfinite(vectors).all()):
        raise ValueError("Invalid label embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError("Invalid label embedding norms")
    return vectors / norms, identity


def label_margin(documents, positive):
    if documents.shape[1] != positive.shape[1]:
        raise ValueError("Document and label embedding dimensions differ")
    similarities = np.asarray(documents, dtype=float) @ np.asarray(positive, dtype=float).T
    if not np.isfinite(similarities).all() or (np.abs(similarities) > 1 + 1e-6).any():
        raise ValueError("Invalid cosine similarity; validate unit-vector embeddings")
    similarities = np.clip(similarities, -1, 1)
    top = np.sort(similarities, axis=1)[:, -2:]
    return bounded(1 - (top[:, 1] - top[:, 0]) / 2), dict(formula="1-(top1-top2)/2")


def logit_indices(model_config, *, entailment=None, contradiction=None):
    mapping = {str(v).lower(): int(k) for k, v in model_config.id2label.items()}
    def find(fragment, supplied):
        found = [index for name, index in mapping.items() if fragment in name]
        if supplied is not None:
            if supplied not in mapping.values():
                raise ValueError("Configured NLI index is outside checkpoint labels")
            return supplied
        if len(found) != 1:
            raise ValueError(f"Ambiguous NLI {fragment} mapping; set an explicit index")
        return found[0]
    e, c = find("entail", entailment), find("contrad", contradiction)
    if e == c:
        raise ValueError("Entailment and contradiction indices must differ")
    return e, c


def from_evidence(evidence, *, multilabel):
    evidence = np.asarray(evidence, dtype=float)
    if not np.isfinite(evidence).all():
        raise ValueError("Nonfinite NLI evidence")
    if multilabel:
        p = expit(evidence)
        probabilities = np.stack((1 - p, p), axis=-1)
        scores = entropy(probabilities).mean(axis=1) / np.log(2)
    else:
        probabilities = softmax(evidence, axis=1)
        scores = entropy(probabilities) / np.log(evidence.shape[1])
    return bounded(scores), probabilities


def predictive_entropy(corpus, hypotheses, config, *, multilabel):
    import gc
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(config.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.nli_model,
        torch_dtype=getattr(torch, config.torch_dtype)).to(config.device).eval()
    e, c = logit_indices(model.config, entailment=config.nli_entailment_index,
                         contradiction=config.nli_contradiction_index)
    hypothesis_ids = [tokenizer(h, add_special_tokens=False).input_ids for h in hypotheses]
    overhead = tokenizer.num_special_tokens_to_add(pair=True)
    capacity = config.nli_max_tokens - overhead - max(map(len, hypothesis_ids))
    max_positions = getattr(model.config, "max_position_embeddings", config.nli_max_tokens)
    if capacity < 1 or config.nli_max_tokens > max_positions:
        raise ValueError("NLI context/hypothesis budget is invalid")
    if config.nli_batch_size < 1:
        raise ValueError("nli_batch_size must be positive")
    evidence = np.zeros((len(corpus.ids), len(hypotheses)), dtype=float)
    denominators = np.zeros(len(corpus.ids), dtype=float)
    buffer, associations, chunk_records = [], [], []
    input_tokens, passes = 0, 0

    def flush():
        nonlocal input_tokens, passes
        if not buffer:
            return
        batch = tokenizer.pad(buffer, padding=True, return_tensors="pt")
        input_tokens += int(batch["attention_mask"].sum().item())
        with torch.inference_mode():
            logits = model(**{k: v.to(config.device) for k, v in batch.items()}).logits.float().cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite NLI logits")
        for row, (document, label, weight) in enumerate(associations):
            evidence[document, label] += weight * float(logits[row, e] - logits[row, c])
        passes += 1
        buffer.clear()
        associations.clear()

    try:
        for document, (identifier, text) in enumerate(zip(corpus.ids, corpus.texts)):
            premise = tokenizer(text, add_special_tokens=False, truncation=False).input_ids
            if not premise:
                raise ValueError("NLI premise must not be empty")
            denominators[document] = len(premise)
            boundaries = []
            for start in range(0, len(premise), capacity):
                chunk = premise[start:start + capacity]
                boundaries.append([start, start + len(chunk)])
                for label, hyp in enumerate(hypothesis_ids):
                    encoded = tokenizer.prepare_for_model(chunk, pair_ids=hyp,
                        add_special_tokens=True, truncation=False, return_attention_mask=True)
                    if len(encoded["input_ids"]) > config.nli_max_tokens:
                        raise ValueError("NLI chunk exceeds declared context")
                    buffer.append(encoded)
                    associations.append((document, label, len(chunk)))
                    if len(buffer) >= config.nli_batch_size:
                        flush()
            chunk_records.append(dict(id=identifier, premise_tokens=len(premise),
                                      boundaries=boundaries, omitted_tokens=0))
        flush()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    evidence /= denominators[:, None]
    scores, _ = from_evidence(evidence, multilabel=multilabel)
    return scores, dict(model=config.nli_model,
        entailment_index=e, contradiction_index=c,
        conversion="token_weighted_mean(entailment_logit-contradiction_logit)",
        probability_rule="independent_sigmoid" if multilabel else "class_softmax_temperature_1",
        neutral_logit="ignored", log_base="e", calibration="none",
        chunking="nonoverlapping_full_text", max_tokens=config.nli_max_tokens,
        dtype=config.torch_dtype, hypotheses=hypotheses, chunks=chunk_records,
        auxiliary_input_tokens=input_tokens, forward_passes=passes)


def prepare(args, config, paths):
    length_tokenizer = args.length_tokenizer or config.length_tokenizer
    if args.count_tokens and not length_tokenizer:
        raise ValueError("Для --count-tokens задайте --length-tokenizer или FEATURES_LENGTH_TOKENIZER")
    if not args.count_tokens and args.length_tokenizer:
        raise ValueError("Параметры токенизатора длин требуют --count-tokens")
    source = read_population(args.dataset, paths["csv"],
        length_tokenizer=length_tokenizer if args.count_tokens else None)
    if args.count_tokens:
        print(f"Документов: {len(source.ids)}; токенов в документах: {sum(source.lengths.values())}",
              file=sys.stderr, flush=True)
    output = paths["output_csv"]
    output_metadata = paths["output_metadata"] or output.with_suffix(".metadata.json")
    if output == paths["csv"] or output_metadata in {paths["csv"], output}:
        raise ValueError("Исходный CSV, выходной CSV и выходные метаданные должны иметь разные пути")
    resource = label_resource(args.dataset)
    frame = source.frame.drop(columns=[column for column in source.frame if column.startswith("diff_")]).copy()
    entries = {}
    selected = METRICS if "all" in args.metrics else tuple(dict.fromkeys(args.metrics))
    documents = None
    for metric in selected:
        column = f"diff_{metric}_raw"
        if metric in {"knn_distance", "label_margin"} and documents is None:
            if not paths["embeddings"] or not paths["embedding_ids"]:
                raise ValueError("Для D2/D3 нужны --embeddings и --embedding-ids")
            documents, embedding_source = load_embeddings(source, paths["embeddings"], paths["embedding_ids"],
                trusted_pickle=args.trusted_pickle_ids, metadata_path=paths["embedding_metadata"], config=config)
        if metric == "flesch_kincaid":
            raw, detail = readability(source.texts)
        elif metric == "ppl":
            raw, detail = perplexity(source.texts, config)
        elif metric == "knn_distance":
            raw, detail = knn_distance(documents, neighbors=config.knn_neighbors,
                                       query_batch_size=config.knn_query_batch_size)
        elif metric == "label_margin":
            positive, encoding = label_embeddings(resource, config,
                paths["label_cache"] or output.parent / "label_embedding_cache")
            raw, detail = label_margin(documents, positive)
            detail["label_encoding"] = encoding
        else:
            raw, detail = predictive_entropy(source, resource["positive"], config,
                                             multilabel=source.spec["task"] == "multilabel")
        if metric in {"knn_distance", "label_margin"}:
            detail["embedding_source"] = embedding_source
        entries[metric] = dict(detail=detail, feature_config=asdict(config),
            computed_at=utc_now(), software=versions(), code=code_hashes())
        frame[column] = validate_scores(metric, raw, len(source.ids))
    atomic_text(output, frame.to_csv(index=False))
    saved = read_population(args.dataset, output)
    for metric, entry in entries.items():
        column = f"diff_{metric}_raw"
        if column in saved.frame:
            values = validate_scores(metric, saved.frame[column], len(saved.ids))
            entry["raw_scores_sha256"] = fingerprint(values.tolist())
    metadata = dict(format=FEATURE_FORMAT, dataset=args.dataset,
        document_ids=list(saved.ids), scoring_content_sha256=saved.manifest["scoring_content_sha256"],
        canonical_lengths_sha256=saved.manifest["canonical_lengths_sha256"],
        source_csv_sha256=source.manifest["csv_sha256"], augmented_csv_sha256=file_hash(output),
        token_length_provenance=source.length_provenance,
        uses_gold_labels=False, uses_target_predictions=False,
        description_resource=resource, metrics=entries, saved_at=utc_now())
    save_json(output_metadata, metadata)
    return dict(csv=str(output), metadata=str(output_metadata),
                token_len_total=sum(saved.lengths.values()), token_len_documents=len(saved.ids))
