from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
K_VALUES = (2, 4, 8, 16, 32, 64)
L_MAX = 50_000
MAX_TOKENS = 7_000
GROUPINGS = ("bfd", "diff_close", "diff_diverse", "diff_routing")
ORDERINGS = ("random", "diff_edges", "diff_disperse")
METRICS = ("flesch_kincaid", "ppl", "knn_distance", "label_margin", "predictive_entropy")
FEATURE_FORMAT = "difficulty-metrics-v1"
MODELS = {"qwen3.5-4b": ("QWEN35", "Qwen3.5-4B", False),
          "ministral3-3b": ("MINISTRAL3", "Ministral3-3B-Instruct", True),
          "gemma4-e4b-it": ("GEMMA4", "Gemma4-E4B-IT", True)}
PARSER_VERSION = "json-fence-document-outcomes-v4"
ERROR_POLICY = "classification_failures_as_errors_v1"
TOKEN_POLICY = "successful_responses_only_v1"
ALGORITHM_VERSION = "raw-score-spans-v1"
DATASETS = {
    "fake_news": dict(task="binary", payload=("title", "text", "date"),
                      scoring=("title", "text"), gold="label"),
    "yahoo": dict(task="multiclass", payload=("title", "content", "answer"),
                  scoring=("title", "content", "answer"), gold="class_label"),
    "goemotions": dict(task="multilabel", payload=("text",), scoring=("text",), gold="class_label"),
}


def load_environment(path=None):
    load_dotenv(path or ROOT / ".env", override=False)


def external_path(value):
    return Path(value).expanduser().resolve() if value else None


def setting(args, name, *, required=False):
    value = getattr(args, name, None) or os.getenv(f"{args.dataset.upper()}_{name.upper()}")
    if name == "output_root":
        value = getattr(args, name, None) or os.getenv("OUTPUT_ROOT")
    if required and not value:
        raise ValueError(f"Укажите --{name.replace('_', '-')} или соответствующую переменную .env")
    return external_path(value)


@dataclass(frozen=True)
class AlgorithmConfig:
    cluster_span: float = 0.20
    batch_span: float = 0.15
    knee_min_strength: float = 0.05
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 100
    kmeans_tol: float = 1e-4
    routing_strength: float = 0.5
    relax_step: float = 0.05
    max_relax_iters: int = 8
    easy_hardness_threshold: float = 0.6
    compact_into_max_hardness: float = 0.6


ALGORITHMS = AlgorithmConfig()


@dataclass(frozen=True)
class FeatureConfig:
    length_tokenizer: str | None = None
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    embedding_max_seq_length: int = 25000
    embedding_padding_side: str = "left"
    embedding_batch_size: int = 8
    ppl_model: str = "Qwen/Qwen3-0.6B"
    ppl_context_tokens: int = 16384
    ppl_stride: int = 512
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_max_tokens: int = 512
    nli_batch_size: int = 16
    nli_entailment_index: int | None = None
    nli_contradiction_index: int | None = None
    knn_neighbors: int = 20
    knn_query_batch_size: int = 256
    device: str = "cuda:0"
    torch_dtype: str = "float32"


@dataclass(frozen=True)
class ModelConfig:
    identity: str
    served_model: str
    structured_output: bool
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = MAX_TOKENS
    context_window: int = 65536
    sampling_seed: int | None = None
    tokenizer_name: str | None = None
    chat_template_path: str | None = None
    tokenizer_match_confirmed: bool = False
    prompt_overhead_tokens: int = 7000
    request_options_json: str = "{}"
    chat_template_kwargs_json: str = "{}"
    concurrency: int = 16
    timeout_seconds: float = 600.0
    transport_attempts: int = 1
    structural_attempts: int = 1
    backoff_initial_seconds: float = 1.0
    backoff_max_seconds: float = 30.0

    def public_dict(self):
        return {key: value for key, value in asdict(self).items()
                if key not in {"base_url", "api_key"}}


def from_environment(cls, prefix, **overrides):
    defaults = cls(**overrides)
    values = asdict(defaults)
    optional_ints = {"sampling_seed", "nli_entailment_index", "nli_contradiction_index"}
    for field in fields(cls):
        raw = os.getenv(f"{prefix}_{field.name.upper()}")
        if raw is None or raw == "" or field.name in overrides:
            continue
        current = values[field.name]
        if isinstance(current, bool):
            if raw.lower() not in {"true", "false"}:
                raise ValueError(f"{prefix}_{field.name.upper()}: требуется true или false")
            values[field.name] = raw.lower() == "true"
        elif isinstance(current, int) or field.name in optional_ints:
            values[field.name] = int(raw)
        elif isinstance(current, float):
            values[field.name] = float(raw)
        else:
            values[field.name] = raw
    return cls(**values)


def embedding_settings(config):
    return {name: value for name, value in asdict(config).items()
            if name.startswith("embedding_") or name == "torch_dtype"}


def model_config(name, *, execute=False):
    prefix, served, structured = MODELS[name]
    model = from_environment(ModelConfig, prefix, identity=name,
                             served_model=os.getenv(f"{prefix}_SERVED_MODEL") or served,
                             structured_output=structured)
    if execute and (not model.base_url or not model.api_key):
        raise ValueError(f"Заполните {prefix}_BASE_URL и {prefix}_API_KEY; для сервера без авторизации задайте EMPTY")
    for value in (model.request_options_json, model.chat_template_kwargs_json):
        if not isinstance(json.loads(value), dict):
            raise ValueError("Настройки запроса и chat template должны быть JSON-объектами")
    return model
