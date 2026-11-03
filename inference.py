import asyncio
from ast import literal_eval
from dataclasses import asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
from pathlib import Path
import uuid

import numpy as np

from batching import GROUPERS, ORDERERS, validate_partition, validate_permutation
from common import code_hashes, fingerprint, read_json, save_json, utc_now, versions, writer_lock
from config import ALGORITHMS, ALGORITHM_VERSION, ERROR_POLICY, L_MAX, MAX_TOKENS, PARSER_VERSION, TOKEN_POLICY
from dataset_io import label_resource
from prompts import parse_predictions, prediction_schema, scoring_view


def validate_model(model):
    if (type(model.max_tokens) is not int or type(model.context_window) is not int
            or model.max_tokens != MAX_TOKENS or model.context_window <= model.max_tokens
            or model.concurrency < 1 or model.timeout_seconds <= 0
            or model.transport_attempts < 1 or model.structural_attempts < 1
            or type(model.prompt_overhead_tokens) is not int or model.prompt_overhead_tokens < 0
            or not math.isfinite(model.temperature)):
        raise ValueError("Invalid model limits/runtime settings")
    options = json.loads(model.request_options_json)
    template_kwargs = json.loads(model.chat_template_kwargs_json)
    if not isinstance(options, dict) or not isinstance(template_kwargs, dict):
        raise ValueError("Request/template settings must be JSON objects")
    if "truncate_prompt_tokens" in json.dumps(options):
        raise ValueError("Server-side silent prompt truncation is forbidden")
    if set(template_kwargs) & {"tokenize", "add_generation_prompt", "return_tensors", "return_dict"}:
        raise ValueError("Chat template kwargs cannot override controlled rendering arguments")
    reserved = {"model", "messages", "temperature", "max_tokens", "max_completion_tokens",
                "max_new_tokens", "max_length", "seed", "stream", "n", "tools", "tool_choice",
                "functions", "function_call", "extra_headers", "api_key"}
    extra = options.get("extra_body", {})
    if not isinstance(extra, dict) or (set(options) | set(extra)) & reserved:
        raise ValueError("Reserved generation controls must use the explicit model fields")
    if model.structured_output and (set(options) | set(extra)) & {
            "response_format", "structured_outputs", "guided_json", "guided_regex",
            "guided_choice", "guided_grammar"}:
        raise ValueError("structured_output controls response constraints")
    server_template = options.get("extra_body", {}).get("chat_template_kwargs", {})
    if server_template and server_template != template_kwargs:
        raise ValueError("Request/local tokenizer template kwargs differ")
    if not model.identity.startswith("qwen") and "enable_thinking" in json.dumps([options, template_kwargs]):
        raise ValueError("Qwen thinking flags cannot be forwarded to another model family")


def completion_kwargs(model, messages, *, output_schema=None):
    options = json.loads(model.request_options_json)
    template_kwargs = json.loads(model.chat_template_kwargs_json)
    if template_kwargs:
        options.setdefault("extra_body", {})["chat_template_kwargs"] = template_kwargs
    result = dict(options, model=model.served_model, messages=messages,
                  temperature=model.temperature, max_tokens=model.max_tokens)
    if model.structured_output:
        if output_schema is None:
            raise ValueError("structured_output requires a per-request output schema")
        result["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "predictions", "strict": True, "schema": output_schema}}
    if model.sampling_seed is not None:
        result["seed"] = model.sampling_seed
    return result


def token_counter(model):
    kwargs = json.loads(model.chat_template_kwargs_json)
    if not model.tokenizer_name:
        return None, dict(tokenizer=None, chat_template_sha256=None,
                          chat_template_kwargs=kwargs, context_budgeting="csv_token_len",
                          prompt_overhead_included=model.prompt_overhead_tokens > 0,
                          prompt_overhead_tokens=model.prompt_overhead_tokens)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.tokenizer_name,
        local_files_only=True, trust_remote_code=False)
    if model.chat_template_path:
        tokenizer.chat_template = Path(model.chat_template_path).expanduser().read_text(encoding="utf-8")
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template; supply chat_template_path")
    identity = dict(tokenizer=model.tokenizer_name,
                    chat_template_sha256=fingerprint(tokenizer.chat_template), chat_template_kwargs=kwargs)

    def count(messages):
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
        if isinstance(tokens, dict):
            tokens = tokens["input_ids"]
        return len(tokens)
    return count, identity


def make_plan(population, model, prompts, *, grouping, ordering, metric, k, seed=42,
              provenance=None, counter=None):
    validate_model(model)
    count, tokenizer_identity = counter if counter is not None else token_counter(model)
    baseline = grouping == "single_doc"
    dependent = not baseline and (grouping != "bfd" or ordering != "random")
    column = f"diff_{metric}_raw" if dependent else None
    scores = dict(zip(population.ids, population.frame[column].astype(float).tolist())) if dependent else {}
    if scores and not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("Difficulty scores must be finite")
    if baseline:
        batches = [{i: population.payloads[i]} for i in population.ids]
    else:
        payload_fields = population.spec["payload"]
        kwargs = dict(id_col="index", len_col="token_len", text_col=payload_fields[0],
                      payload_cols=None if len(payload_fields) == 1 else {f: f for f in payload_fields})
        if grouping != "bfd":
            kwargs.update(diff_col=column, seed=seed)
        batches = GROUPERS[grouping](population.frame, k=k, l=L_MAX, **kwargs)
    original = [list(batch) for batch in batches]
    validate_partition(list(population.ids), original, k, population.lengths, L_MAX)
    ontology = label_resource(population.dataset)["labels"]
    nodes = []
    for index, batch in enumerate(original):
        ordered = batch.copy() if baseline else ORDERERS[ordering](batch, scores=scores,
                    lengths=population.lengths, seed=seed)
        validate_permutation(batch, ordered)
        messages = prompts.messages(ordered, population.payloads)
        canonical = sum(population.lengths[i] for i in ordered)
        node = dict(batch_id=index, grouped_ids=batch, document_ids=ordered,
                    canonical_tokens=canonical, messages=messages,
                    estimated_prompt_tokens=count(messages) if count else canonical + model.prompt_overhead_tokens)
        if model.structured_output:
            node["output_schema"] = prediction_schema(ordered, ontology, population.spec["task"])
        node["request_sha256"] = fingerprint(completion_kwargs(model, messages, output_schema=node.get("output_schema")))
        nodes.append(node)
    validate_partition(list(population.ids), [n["document_ids"] for n in nodes], k, population.lengths, L_MAX)
    condition = dict(schema_version=1, batch_policy="fixed_algorithm_batches_v1",
        dataset=population.dataset, model=model.public_dict(),
        endpoint_sha256=fingerprint(model.base_url), population=population.manifest,
        prompt=prompts.identities, tokenizer=tokenizer_identity, grouping=grouping, ordering=ordering,
        difficulty_metric=metric if dependent else None, difficulty_score_space="raw" if dependent else None,
        difficulty_score_column=column, difficulty_provenance=provenance if dependent else None,
        algorithm_parameters=asdict(ALGORITHMS), algorithm_version=ALGORITHM_VERSION,
        parser_version=PARSER_VERSION, prediction_error_policy=ERROR_POLICY,
        token_accounting_policy=TOKEN_POLICY, algorithm_seed=seed, l_max=L_MAX,
        code_sha256=code_hashes(), software=versions())
    condition_id = fingerprint(condition)
    return dict(condition=condition, condition_id=condition_id, k=k,
                experiment_id=fingerprint([condition_id, k]), batches=nodes)


def gold_labels(value, task, ontology):
    if task == "binary":
        if isinstance(value, (bool, np.bool_)):
            labels = ["true" if value else "false"]
        elif str(value).strip().lower() in {"true", "false", "0", "1", "0.0", "1.0"}:
            labels = ["true" if str(value).strip().lower() in {"true", "1", "1.0"} else "false"]
        else:
            raise ValueError(f"Invalid binary gold label {value!r}")
    elif task == "multilabel":
        if isinstance(value, str):
            value = literal_eval(value) if value.strip().startswith("[") else value.split(",")
        if not isinstance(value, (list, tuple)):
            raise ValueError("Multilabel gold must be a list or comma-delimited exact labels")
        labels = [str(x).strip() for x in value]
    else:
        labels = [str(value).strip()]
    if not labels or len(set(labels)) != len(labels) or not set(labels) <= set(ontology):
        raise ValueError(f"Invalid gold ontology membership: {labels}")
    return labels


def validate_gold(population):
    ontology = label_resource(population.dataset)["labels"]
    column = population.spec["gold"]
    if column not in population.frame:
        raise ValueError(f"Для оценки нужен столбец {column}")
    return [gold_labels(value, population.spec["task"], ontology) for value in population.frame[column]]


def error_category(error):
    status = getattr(error, "status_code", None)
    text = str(error).lower()
    if status in {400, 413, 422} and any(s in text for s in ("context_length", "maximum context", "context window", "too many tokens")):
        return "input_context"
    if status in {408, 409, 429, 500, 502, 503, 504} or any(
            s in type(error).__name__.lower() for s in ("timeout", "connection")):
        return "transient_transport"
    return "configuration_or_transport_fatal"


def retry_delay(error, attempt, model):
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    value = headers.get("retry-after") if headers else None
    if value:
        try:
            return max(0., float(value))
        except ValueError:
            try:
                return max(0., (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError):
                pass
    return min(model.backoff_max_seconds, model.backoff_initial_seconds * 2 ** attempt)


async def classify_batch(client, node, model, ontology, task):
    invalid, transports, requests = 0, 0, 0
    request = completion_kwargs(model, node["messages"], output_schema=node.get("output_schema"))
    while True:
        requests += 1
        try:
            response = await client.chat.completions.create(**request)
        except Exception as exc:
            category = error_category(exc)
            if category == "transient_transport":
                transports += 1
                if transports < model.transport_attempts:
                    delay = min(model.backoff_max_seconds, model.backoff_initial_seconds * 2 ** (transports - 1))
                    await asyncio.sleep(max(delay, retry_delay(exc, 0, model)))
                    continue
            elif category == "input_context":
                invalid += 1
                if invalid < model.structural_attempts:
                    continue
            raise RuntimeError(f"Батч {node['batch_id']}: {category}") from None
        raw = response.model_dump() if hasattr(response, "model_dump") else response
        if not isinstance(raw, dict):
            raise TypeError("Unserializable completion container")
        first = (raw.get("choices") or [{}])[0]
        message = first.get("message") or {}
        content = message.get("content") or ""
        finish = first.get("finish_reason")
        _, flags, error = parse_predictions(content, node["document_ids"], ontology, task)
        if finish == "length":
            category = "output_truncation"
        elif finish != "stop" or message.get("refusal") or not all(
                value for name, value in flags.items() if name != "labels_valid"):
            category = "structural_output"
        elif not flags["labels_valid"]:
            category = "classification_error"
        else:
            category = "valid"
        if category in {"output_truncation", "structural_output"}:
            invalid += 1
            if invalid < model.structural_attempts:
                continue
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            raise ValueError(f"Батч {node['batch_id']}: ответ без usage")
        incoming, outgoing = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if type(incoming) is not int or type(outgoing) is not int or incoming <= 0 or outgoing < 0:
            raise ValueError(f"Батч {node['batch_id']}: некорректный usage")
        view = scoring_view(content, node["document_ids"], ontology, task)
        return dict(view, batch_id=node["batch_id"], document_ids=node["document_ids"],
                    grouped_ids=node["grouped_ids"], request_sha256=node["request_sha256"],
                    canonical_tokens=node["canonical_tokens"], estimated_prompt_tokens=node["estimated_prompt_tokens"],
                    response_id=raw.get("id"), category=category, validation=flags, validation_error=error,
                    raw_content=content, finish_reason=finish, tokens_in=incoming, tokens_out=outgoing,
                    requests_sent=requests)


async def execute(plan, population, model, directory):
    from openai import AsyncOpenAI
    from sklearn.metrics import f1_score
    if not model.base_url or not model.api_key:
        raise ValueError("Для инференса нужны endpoint и API-ключ")
    ontology = label_resource(population.dataset)["labels"]
    gold = validate_gold(population)
    run_id, started = uuid.uuid4().hex, utc_now()
    nodes = iter(plan["batches"])
    responses = [None] * len(plan["batches"])

    with writer_lock(directory):
        result_path = Path(directory) / "result.json"
        if result_path.exists():
            previous = read_json(result_path)
            if (previous.get("condition_id") != plan["condition_id"]
                    or previous.get("k") != plan["k"]
                    or previous.get("parser_version") != PARSER_VERSION
                    or previous.get("prediction_error_policy") != ERROR_POLICY
                    or previous.get("token_accounting_policy") != TOKEN_POLICY):
                raise ValueError("В каталоге сохранено другое условие или версия правил; выберите новый output-root")
        async with AsyncOpenAI(base_url=model.base_url, api_key=model.api_key,
                              timeout=model.timeout_seconds, max_retries=0) as client:
            async def worker():
                for node in nodes:
                    responses[node["batch_id"]] = await classify_batch(
                        client, node, model, ontology, population.spec["task"])
            async with asyncio.TaskGroup() as group:
                for _ in range(min(model.concurrency, len(responses))):
                    group.create_task(worker())
        predictions, failures = {}, []
        for response in responses:
            predictions.update(response["predictions"])
            failures.extend(response["failures"])
        if set(predictions) != set(population.ids):
            raise ValueError("Получены исходы не для всех документов")
        true_matrix = [[int(c in labels) for c in ontology] for labels in gold]
        pred_matrix = [[int(c in predictions[i]) for c in ontology] for i in population.ids]
        incoming = sum(r["tokens_in"] for r in responses)
        outgoing = sum(r["tokens_out"] for r in responses)
        result = dict(schema_version=1, execution_status="COMPLETE", condition_id=plan["condition_id"], experiment_id=plan["experiment_id"],
            run_id=run_id, started_at=started, completed_at=utc_now(), condition=plan["condition"], k=plan["k"],
            parser_version=PARSER_VERSION, prediction_error_policy=ERROR_POLICY, token_accounting_policy=TOKEN_POLICY,
            plan_sha256=fingerprint(plan), predictions=predictions,
            macro_f1=float(f1_score(true_matrix, pred_matrix, average="macro", zero_division=0)),
            prediction_coverage=1.0, classification_failures=failures, classification_failure_count=len(failures),
            invalid_prediction_count=len(failures), classification_failure_proportion=len(failures) / len(population.ids),
            tokens_in=incoming, tokens_out=outgoing, total_tokens=incoming + outgoing,
            requests_sent=sum(r["requests_sent"] for r in responses), number_of_batches=len(responses),
            batches=responses)
        save_json(result_path, result)
    return dict(experiment_id=plan["experiment_id"], run_id=run_id, macro_f1=result["macro_f1"],
                total_tokens=result["total_tokens"], result=str(Path(directory) / "result.json"))
