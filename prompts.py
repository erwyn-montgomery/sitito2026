import json
import re
from dataclasses import dataclass

from common import file_hash, fingerprint, read_json
from config import ERROR_POLICY

SYSTEM_PROMPT = (
    "Return ONLY one valid JSON object, with every input ID exactly once. "
    "Use JSON booleans for binary tasks, one exact label string for multiclass "
    "tasks, and a nonempty array of unique exact label strings for multilabel "
    "tasks, as specified by the task. No prose, markdown, or explanations."
)


@dataclass(frozen=True)
class PromptResources:
    task: str
    examples: dict | None
    identities: dict

    @classmethod
    def load(cls, prompt_path, few_shot_path, ids):
        task = prompt_path.read_text(encoding="utf-8")
        examples = read_json(few_shot_path) if few_shot_path else None
        if examples is not None:
            if (not isinstance(examples, dict) or not isinstance(examples.get("in"), dict)
                    or not isinstance(examples.get("out"), dict)
                    or set(examples["in"]) != set(examples["out"])):
                raise ValueError("Few-shot resource requires aligned in/out objects")
            if set(map(str, examples["in"])) & set(ids):
                raise ValueError("Few-shot IDs overlap the evaluation population")
        return cls(task, examples, {"prompt_sha256": file_hash(prompt_path),
            "few_shot_sha256": file_hash(few_shot_path) if few_shot_path else None,
            "system_prompt_sha256": fingerprint(SYSTEM_PROMPT)})

    def messages(self, ids, payloads):
        content = self.task
        if self.examples is not None:
            content += ("\n\nFEW-SHOT EXAMPLES (JSON):\nInput example:\n"
                        + json.dumps(self.examples["in"], ensure_ascii=False, indent=2)
                        + "\n\nOutput example:\n"
                        + json.dumps(self.examples["out"], ensure_ascii=False, indent=2)
                        + "\n\nEND OF FEW-SHOT EXAMPLES.")
        content += ("\n\nCURRENT REQUEST:\nClassify only the INPUT DATA below. "
                    "Return every ID in this list exactly once, and no other IDs: "
                    + json.dumps(list(ids), ensure_ascii=False)
                    + ". Do not copy IDs or predictions from the examples. "
                    "Return the JSON object without markdown fences.")
        content += "\n\nINPUT DATA:\n" + json.dumps({i: payloads[i] for i in ids},
                                                    ensure_ascii=False, indent=2, allow_nan=False)
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def prediction_schema(ids, ontology, task):
    if task == "binary":
        value = {"type": "boolean"}
    elif task == "multiclass":
        value = {"type": "string", "enum": list(ontology)}
    elif task == "multilabel":
        value = {"type": "array", "items": {"type": "string", "enum": list(ontology)},
                 "minItems": 1, "maxItems": len(ontology)}
    else:
        raise ValueError(f"Unsupported task: {task}")
    return {"type": "object", "properties": {i: value.copy() for i in ids},
            "required": list(ids), "additionalProperties": False}


def unwrap_json_fence(content):
    if isinstance(content, str):
        match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", content.strip(),
                             flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
    return content


def label_shape_valid(labels, task):
    return (task in {"binary", "multiclass", "multilabel"}
            and isinstance(labels, list) and bool(labels)
            and all(isinstance(label, str) and bool(label.strip()) for label in labels)
            and len(labels) == len(set(labels))
            and (task == "multilabel" or len(labels) == 1))


def parse_predictions(content, ids, ontology, task):
    duplicate = []
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicate.append(key)
            result[key] = value
        return result
    flags = {"json_parsed": False, "expected_ids_present": False, "no_unexpected_ids": False,
             "no_duplicated_ids": False, "label_shape_valid": False, "labels_valid": False}
    try:
        parsed = json.loads(unwrap_json_fence(content), object_pairs_hook=pairs_hook,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, TypeError) as exc:
        return {}, flags, f"Invalid JSON: {exc}"
    flags["json_parsed"] = True
    flags["no_duplicated_ids"] = not duplicate
    if not isinstance(parsed, dict):
        return {}, flags, "Expected a JSON object"
    flags["expected_ids_present"] = set(ids) <= set(parsed)
    flags["no_unexpected_ids"] = set(parsed) <= set(ids)
    result = {}
    for identifier in ids:
        value = parsed.get(identifier)
        if task == "binary":
            if type(value) is bool:
                labels = ["true" if value else "false"]
            elif isinstance(value, str):
                labels = [value]
            else:
                continue
        elif task == "multiclass":
            if not isinstance(value, str):
                continue
            labels = [value]
        else:
            if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
                continue
            labels = value
        if label_shape_valid(labels, task):
            result[identifier] = labels
    flags["label_shape_valid"] = len(result) == len(ids)
    flags["labels_valid"] = flags["label_shape_valid"] and all(
        set(labels) <= set(ontology) for labels in result.values())
    if all(flags.values()):
        error = None
    elif all(value for name, value in flags.items() if name != "labels_valid"):
        error = "Unknown label strings outside the fixed ontology"
    else:
        error = "Invalid IDs, duplicate keys, or label shapes"
    return result, flags, error


def scoring_view(content, ids, ontology, task):
    if task not in {"binary", "multiclass", "multilabel"}:
        raise ValueError(f"Unsupported task: {task}")
    ids = list(ids)
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) for i in ids):
        raise ValueError("Request membership must contain unique string IDs")
    duplicates = set()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                duplicates.add(key)
            result[key] = value
        return result
    malformed = False
    try:
        parsed = json.loads(unwrap_json_fence(content), object_pairs_hook=pairs,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError):
        parsed, malformed = {}, True
    if not isinstance(parsed, dict):
        parsed, malformed = {}, True
    allowed, predictions, failures = set(ontology), {}, []
    for identifier in ids:
        raw = parsed.get(identifier)
        reasons = []
        if malformed:
            reasons.append("malformed_response")
        elif identifier in duplicates:
            reasons.append("duplicate_document_id")
        elif identifier not in parsed:
            reasons.append("missing_prediction")
        labels = []
        if not reasons:
            if task == "multilabel":
                if isinstance(raw, list):
                    labels = raw
                else:
                    reasons.append("unusable_label_shape")
            elif task == "binary" and type(raw) is bool:
                labels = ["true" if raw else "false"]
            elif isinstance(raw, str):
                labels = [raw]
            else:
                reasons.append("unusable_label_shape")
            if task != "multilabel" and len(labels) != 1:
                reasons.append("unusable_label_shape")
            if not labels:
                reasons.append("no_usable_label")
        known = []
        if not any(r in reasons for r in ("malformed_response", "duplicate_document_id", "missing_prediction", "unusable_label_shape")):
            for label in labels:
                if not isinstance(label, str) or not label.strip():
                    reasons.append("unusable_label")
                elif label not in allowed:
                    reasons.append("hallucinated_label")
                elif label not in known:
                    known.append(label)
        if task != "multilabel" and reasons:
            known = []
        predictions[identifier] = known
        if reasons:
            failures.append(dict(document_id=identifier, failure_categories=sorted(set(reasons)),
                                 raw_prediction=raw, scoring_prediction=known, policy_version=ERROR_POLICY))
    return dict(predictions=predictions, failures=failures, unexpected_ids=sorted(set(parsed) - set(ids)))
