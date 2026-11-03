from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from common import file_hash, fingerprint, read_json
from config import DATASETS, FEATURE_FORMAT, ROOT, embedding_settings


def label_resource(dataset):
    document = read_json(ROOT / "resources" / "label_descriptions.json")
    item = document[dataset]
    labels = item["labels"]
    if (len(labels) != len(set(labels)) or set(labels) != set(item["definitions"])
            or len(labels) != {"fake_news": 2, "yahoo": 10, "goemotions": 28}[dataset]):
        raise ValueError("Некорректная онтология классов")
    positive = [item["template"].format(label=label, definition=item["definitions"][label])
                for label in labels]
    result = dict(item, positive=positive, version=document["version"])
    return dict(result, sha256=fingerprint(result))


def read_corpus(dataset, path):
    path = Path(path)
    spec = DATASETS[dataset]
    frame = pd.read_csv(path, dtype={"index": str}, keep_default_na=False)
    required = {"index", *spec["scoring"]}
    if frame.empty or required - set(frame):
        raise ValueError(f"Пустой CSV или отсутствуют поля: {sorted(required - set(frame))}")
    ids = tuple(frame["index"].astype(str))
    if any(not i or i.strip() != i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("index: нужны уникальные непустые строковые ID без пробелов по краям")
    texts = tuple("\n\n".join(str(row[f]).strip() for f in spec["scoring"] if str(row[f]).strip())
                  for row in frame.to_dict("records"))
    if any(not text for text in texts):
        raise ValueError("Пустой текст документа")
    return SimpleNamespace(dataset=dataset, frame=frame, ids=ids, texts=texts)


def read_population(dataset, path, *, length_tokenizer=None):
    corpus = read_corpus(dataset, path)
    frame, ids, texts = corpus.frame, corpus.ids, corpus.texts
    spec = DATASETS[dataset]
    required = set(spec["payload"])
    if not length_tokenizer:
        required.add("token_len")
    if required - set(frame):
        raise ValueError(f"Отсутствуют поля: {sorted(required - set(frame))}")
    length_provenance = dict(method="csv_column", column="token_len")
    if length_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(length_tokenizer,
                                                  local_files_only=True, trust_remote_code=False)
        length_texts = ["\n\n".join(str(row[f]).strip() for f in spec["payload"] if str(row[f]).strip())
                        for row in frame.to_dict("records")]
        frame["token_len"] = [len(tokenizer(text, add_special_tokens=False, truncation=False).input_ids)
                              for text in length_texts]
        length_provenance = dict(method="classification_fields_joined_double_newline",
            fields=list(spec["payload"]), tokenizer=length_tokenizer,
            add_special_tokens=False, truncation=False)
    lengths = pd.to_numeric(frame["token_len"], errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(lengths).all() or (lengths <= 0).any()
            or (lengths != np.floor(lengths)).any()):
        raise ValueError("token_len: нужны конечные положительные целые длины")
    payloads = {}
    for identifier, row in zip(ids, frame.to_dict("records")):
        if any(not isinstance(row[f], str) for f in spec["payload"]):
            for f in spec["payload"]:
                row[f] = str(row[f])
        payloads[identifier] = (row[spec["payload"][0]] if len(spec["payload"]) == 1
                                else {f: row[f] for f in spec["payload"]})
    resource = label_resource(dataset)
    manifest = dict(dataset=dataset, csv_sha256=file_hash(path), document_ids=list(ids),
                    ordered_ids_sha256=fingerprint(ids),
                    scoring_content_sha256=fingerprint(list(zip(ids, texts))),
                    payload_sha256=fingerprint([(i, payloads[i]) for i in ids]),
                    canonical_lengths_sha256=fingerprint(lengths.astype(int).tolist()),
                    ontology_sha256=fingerprint(resource["labels"]), schema=spec)
    return SimpleNamespace(dataset=dataset, spec=spec, frame=frame, ids=ids, texts=tuple(texts),
                           lengths=dict(zip(ids, lengths.astype(int).tolist())),
                           payloads=payloads, manifest=manifest, length_provenance=length_provenance)


def validate_scores(metric, values, count):
    values = np.asarray(values, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f"{metric}: нужны конечные оценки сложности для всех ID")
    if metric == "ppl" and (values <= 0).any():
        raise ValueError("PPL должна быть положительной")
    if metric == "knn_distance" and ((values < 0).any() or (values > 2).any()):
        raise ValueError("Косинусное расстояние должно принадлежать [0,2]")
    if metric in {"label_margin", "predictive_entropy"} and ((values < 0).any() or (values > 1).any()):
        raise ValueError(f"{metric}: значения должны принадлежать [0,1]")
    return values


def load_scores(population, metric, metadata_path):
    column = f"diff_{metric}_raw"
    if column not in population.frame:
        raise ValueError(f"Нет {column}; сначала вычислите метрику командой prepare")
    if not metadata_path:
        raise ValueError("Укажите --metadata с файлом, созданным командой prepare")
    raw = validate_scores(metric, pd.to_numeric(population.frame[column], errors="raise"), len(population.ids))
    metadata = read_json(metadata_path)
    if metadata.get("format") != FEATURE_FORMAT or metadata.get("dataset") != population.dataset:
        raise ValueError("Нужны метаданные этого датасета, созданные командой prepare")
    for name in ("document_ids", "scoring_content_sha256", "canonical_lengths_sha256"):
        if metadata.get(name) != population.manifest[name]:
            raise ValueError(f"Метрики не совпадают с CSV: {name}")
    if metadata.get("augmented_csv_sha256") != population.manifest["csv_sha256"]:
        raise ValueError("CSV изменён после подготовки метрик")
    measurement = metadata.get("metrics", {}).get(metric)
    if not measurement or not measurement.get("computed_at") or not measurement.get("feature_config"):
        raise ValueError(f"Сначала вычислите {metric} командой prepare")
    if measurement.get("raw_scores_sha256") != fingerprint(raw.tolist()):
        raise ValueError(f"Не совпадает хэш метрики: {metric}")
    if metric in {"label_margin", "predictive_entropy"}:
        resource = label_resource(population.dataset)
        if metadata.get("description_resource") != resource:
            raise ValueError("Канонические описания отличаются от использованных при расчёте")
    return raw, dict(measurement=measurement, metadata_sha256=file_hash(metadata_path),
                     token_length_provenance=metadata.get("token_length_provenance"))


def load_embeddings(population, path, ids_path, *, trusted_pickle=False, metadata_path=None, config=None):
    path = Path(path)
    ids_path = Path(ids_path)
    saved = read_json(ids_path) if ids_path.suffix == ".json" else np.load(ids_path, allow_pickle=trusted_pickle)
    if isinstance(saved, np.ndarray):
        if saved.ndim != 1:
            raise ValueError("ID эмбеддингов должны быть одномерными")
        saved = saved.tolist()
    if not isinstance(saved, list) or any(isinstance(i, (list, dict)) for i in saved):
        raise ValueError("Нужен одномерный список ID эмбеддингов")
    ids = [str(i) for i in saved]
    if len(ids) != len(set(ids)) or set(ids) != set(population.ids):
        raise ValueError("ID эмбеддингов не совпадают с полным корпусом CSV")
    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] < 1:
        raise ValueError("Неверная размерность документных эмбеддингов")
    positions = {identifier: i for i, identifier in enumerate(ids)}
    x = np.asarray(vectors[[positions[i] for i in population.ids]], dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or not np.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError("Некорректные или нулевые эмбеддинги")
    source = dict(embeddings_sha256=file_hash(path), ids_file_sha256=file_hash(ids_path),
                  generation_provenance="not_verified")
    manifest_path = Path(metadata_path) if metadata_path else path.with_suffix(".metadata.json")
    if metadata_path or manifest_path.is_file():
        manifest = read_json(manifest_path)
        expected = dict(format="document-embeddings-v1", dataset=population.dataset,
            embeddings_sha256=source["embeddings_sha256"], ids_file_sha256=source["ids_file_sha256"],
            document_ids=ids, shape=list(vectors.shape),
            score_text_fields=list(population.spec["scoring"]), normalize_embeddings=True, prompt="",
            scoring_content_by_id_sha256=fingerprint(sorted(zip(population.ids, population.texts))))
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Эмбеддинги, ID или тексты отличаются от использованных при кодировании")
        if config is not None and manifest.get("embedding_config") != embedding_settings(config):
            raise ValueError("Настройки кодирования документов и метрик должны совпадать")
        source.update(generation_provenance="recorded", metadata_sha256=file_hash(manifest_path),
                      embedding_config=manifest["embedding_config"])
    return x / norms, source
