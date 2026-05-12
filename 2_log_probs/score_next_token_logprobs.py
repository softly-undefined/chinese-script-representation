from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False


@dataclass
class CandidateRequest:
    event_index: int
    candidate_name: str
    candidate_text: str
    input_ids: list[int]
    candidate_positions: list[int]
    candidate_token_ids: list[int]
    candidate_tokens: list[str]
    tokenization_mode: str
    prefix_tokens: int
    prefix_tokens_truncated: bool
    bos_added: bool


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def output_root() -> Path:
    return script_dir() / "out"


def ensure_under_output_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed_root = output_root().resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Output path must be inside {allowed_root}: {path}") from exc
    return resolved


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with io.open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            if not isinstance(payload, dict):
                raise TypeError(f"Expected JSON object on line {line_number} in {path}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()


def event_key(event: dict[str, Any], fallback_index: int) -> str:
    event_id = event.get("event_id")
    if isinstance(event_id, str) and event_id:
        return event_id
    return "|".join(
        [
            str(event.get("source_file", "")),
            str(event.get("line_index", "")),
            str(event.get("char_index", "")),
            str(event.get("context_script", "")),
            str(event.get("simplified", "")),
            str(event.get("traditional", "")),
            str(fallback_index),
        ]
    )


def load_completed_event_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with io.open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Existing output has invalid JSON on line {line_number}: {path}"
                ) from exc
            if not isinstance(payload, dict):
                continue
            completed.add(event_key(payload, fallback_index=line_number - 1))
    return completed


def prepare_events_for_scoring(
    events: list[dict[str, Any]],
    *,
    out_jsonl: Path,
    resume: bool,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], int]:
    if overwrite and out_jsonl.exists():
        out_jsonl.unlink()
    if overwrite:
        return events, 0

    completed = load_completed_event_keys(out_jsonl) if resume else set()
    if not completed:
        return events, 0

    pending: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event_key(event, fallback_index=index) not in completed:
            pending.append(event)
    return pending, len(events) - len(pending)


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def stable_random_float(base_seed: int, *parts: object) -> float:
    return stable_seed(base_seed, *parts) / float(2**64 - 1)


def pick_device(requested: str):
    import torch

    req = requested.strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def should_add_bos(mode: str, tokenizer) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        if tokenizer.bos_token_id is None:
            raise ValueError("--add-bos always requested, but tokenizer has no bos_token_id")
        return True
    return tokenizer.bos_token_id is not None


def continuation_ids(prefix: str, candidate: str, tokenizer) -> tuple[list[int], list[int], str]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + candidate, add_special_tokens=False)
    if len(full_ids) >= len(prefix_ids) and full_ids[: len(prefix_ids)] == prefix_ids:
        return prefix_ids, full_ids[len(prefix_ids) :], "prefix_delta"

    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    return prefix_ids, candidate_ids, "standalone_append_after_prefix_mismatch"


def build_candidate_request(
    *,
    event_index: int,
    event: dict[str, Any],
    candidate_name: str,
    candidate_text: str,
    tokenizer,
    add_bos: bool,
    max_prefix_tokens: int | None,
) -> CandidateRequest:
    prefix = event.get("prefix", "")
    if not isinstance(prefix, str):
        raise TypeError(f"Event {event_index} has non-string prefix")

    prefix_ids, candidate_ids, tokenization_mode = continuation_ids(prefix, candidate_text, tokenizer)
    prefix_tokens_truncated = False
    if max_prefix_tokens is not None and len(prefix_ids) > max_prefix_tokens:
        prefix_ids = prefix_ids[-max_prefix_tokens:]
        prefix_tokens_truncated = True

    if not candidate_ids:
        raise ValueError(
            f"Candidate {candidate_name}={candidate_text!r} for event {event_index} tokenized to no ids"
        )

    input_ids: list[int] = []
    if add_bos:
        input_ids.append(int(tokenizer.bos_token_id))
    input_ids.extend(prefix_ids)
    candidate_start = len(input_ids)
    input_ids.extend(candidate_ids)

    if candidate_start == 0:
        raise ValueError(
            f"Event {event_index} has no prefix tokens and no BOS; cannot score first token"
        )

    candidate_positions = list(range(candidate_start, candidate_start + len(candidate_ids)))
    candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_ids)
    return CandidateRequest(
        event_index=event_index,
        candidate_name=candidate_name,
        candidate_text=candidate_text,
        input_ids=input_ids,
        candidate_positions=candidate_positions,
        candidate_token_ids=[int(token_id) for token_id in candidate_ids],
        candidate_tokens=[str(token) for token in candidate_tokens],
        tokenization_mode=tokenization_mode,
        prefix_tokens=len(prefix_ids),
        prefix_tokens_truncated=prefix_tokens_truncated,
        bos_added=add_bos,
    )


def score_candidate_requests(
    requests: list[CandidateRequest],
    *,
    tokenizer,
    model,
    device,
) -> list[float]:
    import torch
    import torch.nn.functional as F

    if not requests:
        return []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            pad_token_id = tokenizer.eos_token_id
        else:
            pad_token_id = 0

    max_len = max(len(request.input_ids) for request in requests)
    input_ids = []
    attention_mask = []
    for request in requests:
        padding = max_len - len(request.input_ids)
        input_ids.append(request.input_ids + [int(pad_token_id)] * padding)
        attention_mask.append([1] * len(request.input_ids) + [0] * padding)

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=attention_tensor)
        log_probs = F.log_softmax(outputs.logits, dim=-1)

    scores: list[float] = []
    for batch_index, request in enumerate(requests):
        total = 0.0
        for position, token_id in zip(request.candidate_positions, request.candidate_token_ids):
            total += float(log_probs[batch_index, position - 1, token_id].detach().cpu().item())
        scores.append(total)
    return scores


def load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    ).to(device)
    model.eval()
    return tokenizer, model, device


def fake_token_ids_for_char(ch: str, seed: int) -> tuple[list[int], list[str]]:
    split_probability = stable_random_float(seed, "fake_token_split", ch)
    token_count = 2 if split_probability < 0.12 else 1
    token_ids: list[int] = []
    tokens: list[str] = []
    for index in range(token_count):
        token_id = 10_000 + stable_seed(seed, "fake_token_id", ch, index) % 900_000
        token_ids.append(int(token_id))
        if token_count == 1:
            tokens.append(f"fake:{ch}")
        else:
            tokens.append(f"fake:{ch}:{index + 1}")
    return token_ids, tokens


def fake_margin_for_event(event: dict[str, Any], args: argparse.Namespace) -> float:
    context_script = str(event.get("context_script", ""))
    mapping_kind = str(event.get("mapping_kind", ""))
    simplified = str(event.get("simplified", ""))
    traditional = str(event.get("traditional", ""))
    event_id = str(event.get("event_id", event.get("line_index", "")))

    if mapping_kind == "opencc_one_to_one_control":
        base_margin = args.fake_control_margin
    elif "many" in mapping_kind:
        base_margin = args.fake_ambiguous_margin
    else:
        base_margin = args.fake_script_margin

    if context_script == "hant":
        script_direction = 1.0
    elif context_script == "hans":
        script_direction = -1.0
    else:
        script_direction = 0.0

    prefix_len = len(str(event.get("prefix", "")))
    prefix_factor = 0.55 + 0.45 * min(prefix_len / 80.0, 1.0)

    script_counts = event.get("script_counts", {})
    mixed_script = isinstance(script_counts, dict) and bool(
        script_counts.get("mixed_script_by_indicators")
    )
    mixed_factor = args.fake_mixed_script_shrink if mixed_script else 1.0

    pair_rng = random.Random(
        stable_seed(args.fake_seed, "pair_bias", simplified, traditional)
    )
    event_rng = random.Random(
        stable_seed(
            args.fake_seed,
            "event_noise",
            event_id,
            event.get("line_index", ""),
            event.get("char_index", ""),
            simplified,
            traditional,
        )
    )

    pair_bias = pair_rng.gauss(0.0, args.fake_pair_bias_sd)
    event_noise = event_rng.gauss(0.0, args.fake_noise)
    return script_direction * base_margin * prefix_factor * mixed_factor + pair_bias + event_noise


def fake_score_event(
    event: dict[str, Any],
    *,
    args: argparse.Namespace,
    model_name: str,
) -> dict[str, Any]:
    simplified = str(event.get("simplified", ""))
    traditional = str(event.get("traditional", ""))
    if not simplified or not traditional:
        out = dict(event)
        out["model_name"] = model_name
        out["score_mode"] = "fake"
        out["score_error"] = "missing_simplified_or_traditional"
        return out

    simplified_ids, simplified_tokens = fake_token_ids_for_char(simplified, args.fake_seed)
    traditional_ids, traditional_tokens = fake_token_ids_for_char(traditional, args.fake_seed)

    event_id = str(event.get("event_id", event.get("line_index", "")))
    rng = random.Random(
        stable_seed(
            args.fake_seed,
            "fake_center_logprob",
            event_id,
            event.get("line_index", ""),
            event.get("char_index", ""),
            simplified,
            traditional,
        )
    )
    delta_before_token_penalty = fake_margin_for_event(event, args)
    prefix_len = len(str(event.get("prefix", "")))
    prefix_tokens = prefix_len
    prefix_tokens_truncated = False
    if args.max_prefix_tokens is not None and prefix_tokens > args.max_prefix_tokens:
        prefix_tokens = args.max_prefix_tokens
        prefix_tokens_truncated = True

    center = -5.5 - 0.006 * min(prefix_len, 240) + rng.gauss(0.0, 0.65)
    simplified_penalty = 0.42 * (len(simplified_ids) - 1)
    traditional_penalty = 0.42 * (len(traditional_ids) - 1)
    simplified_logprob = center - (delta_before_token_penalty / 2.0) - simplified_penalty
    traditional_logprob = center + (delta_before_token_penalty / 2.0) - traditional_penalty

    out = dict(event)
    out.update(
        {
            "model_name": model_name,
            "score_mode": "fake",
            "fake_parameters": {
                "fake_seed": args.fake_seed,
                "fake_script_margin": args.fake_script_margin,
                "fake_ambiguous_margin": args.fake_ambiguous_margin,
                "fake_control_margin": args.fake_control_margin,
                "fake_noise": args.fake_noise,
                "fake_pair_bias_sd": args.fake_pair_bias_sd,
                "fake_mixed_script_shrink": args.fake_mixed_script_shrink,
            },
            "simplified_logprob": simplified_logprob,
            "traditional_logprob": traditional_logprob,
            "delta_logp": traditional_logprob - simplified_logprob,
            "simplified_token_ids": simplified_ids,
            "traditional_token_ids": traditional_ids,
            "simplified_tokens": simplified_tokens,
            "traditional_tokens": traditional_tokens,
            "simplified_num_tokens": len(simplified_ids),
            "traditional_num_tokens": len(traditional_ids),
            "single_token_pair": len(simplified_ids) == 1 and len(traditional_ids) == 1,
            "simplified_tokenization_mode": "fake_tokenization",
            "traditional_tokenization_mode": "fake_tokenization",
            "prefix_tokens": prefix_tokens,
            "prefix_tokens_truncated": prefix_tokens_truncated,
            "bos_added": False,
        }
    )
    return out


def fake_score_events(
    events: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    model_name: str,
) -> list[dict[str, Any]]:
    iterator = events
    if _TQDM:
        iterator = tqdm(events, desc="fake_score", unit="event")
    return [fake_score_event(event, args=args, model_name=model_name) for event in iterator]


def fake_score_incremental(
    events: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    model_name: str,
    out_jsonl: Path,
) -> int:
    iterator = range(0, len(events), args.batch_size)
    if _TQDM:
        iterator = tqdm(
            iterator,
            total=(len(events) + args.batch_size - 1) // args.batch_size,
            desc="fake_score",
            unit="batch",
        )
    written = 0
    for start in iterator:
        batch_events = events[start : start + args.batch_size]
        batch_scored = [
            fake_score_event(event, args=args, model_name=model_name)
            for event in batch_events
        ]
        append_jsonl(out_jsonl, batch_scored)
        written += len(batch_scored)
    return written


def annotate_scores(
    events: list[dict[str, Any]],
    requests: list[CandidateRequest],
    scores: list[float],
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    by_event: dict[int, dict[str, tuple[CandidateRequest, float]]] = {}
    for request, score in zip(requests, scores):
        by_event.setdefault(request.event_index, {})[request.candidate_name] = (request, score)

    scored_events: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        out = dict(event)
        out["model_name"] = model_name
        candidates = by_event.get(event_index, {})
        simplified = candidates.get("simplified")
        traditional = candidates.get("traditional")

        if simplified is None or traditional is None:
            out["score_error"] = "missing_candidate_score"
            scored_events.append(out)
            continue

        simplified_request, simplified_logprob = simplified
        traditional_request, traditional_logprob = traditional
        out.update(
            {
                "simplified_logprob": simplified_logprob,
                "traditional_logprob": traditional_logprob,
                "delta_logp": traditional_logprob - simplified_logprob,
                "simplified_token_ids": simplified_request.candidate_token_ids,
                "traditional_token_ids": traditional_request.candidate_token_ids,
                "simplified_tokens": simplified_request.candidate_tokens,
                "traditional_tokens": traditional_request.candidate_tokens,
                "simplified_num_tokens": len(simplified_request.candidate_token_ids),
                "traditional_num_tokens": len(traditional_request.candidate_token_ids),
                "single_token_pair": len(simplified_request.candidate_token_ids) == 1
                and len(traditional_request.candidate_token_ids) == 1,
                "simplified_tokenization_mode": simplified_request.tokenization_mode,
                "traditional_tokenization_mode": traditional_request.tokenization_mode,
                "prefix_tokens": max(
                    simplified_request.prefix_tokens,
                    traditional_request.prefix_tokens,
                ),
                "prefix_tokens_truncated": simplified_request.prefix_tokens_truncated
                or traditional_request.prefix_tokens_truncated,
                "bos_added": simplified_request.bos_added or traditional_request.bos_added,
            }
        )
        scored_events.append(out)
    return scored_events


def parse_args() -> argparse.Namespace:
    out = output_root()
    parser = argparse.ArgumentParser(
        description=(
            "Score simplified vs traditional candidate continuations with a Hugging Face causal LM. "
            "This computes candidate log probabilities only; it does not generate text."
        )
    )
    parser.add_argument(
        "--events",
        default=str(out / "ambiguity_events.jsonl"),
        help="Input ambiguity events JSONL.",
    )
    parser.add_argument(
        "--out-jsonl",
        default=str(out / "scored_logprobs.jsonl"),
        help=(
            "Output scored JSONL path. Must be inside 2_log_probs/out/. "
            "Existing rows are treated as checkpoints unless --overwrite-output is used."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face model name or local model path. Required unless --fake-scores is used.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto/cpu/cuda/mps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of events to score per model batch.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model only from local cache.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders.",
    )
    parser.add_argument(
        "--add-bos",
        choices=["auto", "always", "never"],
        default="auto",
        help="Whether to prepend tokenizer.bos_token_id before the prefix.",
    )
    parser.add_argument(
        "--max-prefix-tokens",
        type=int,
        default=None,
        help="Optionally left-truncate prefix tokens before candidate scoring.",
    )
    parser.add_argument(
        "--limit-events",
        type=int,
        default=None,
        help="Optional event limit for quick scoring checks.",
    )
    parser.add_argument(
        "--fake-scores",
        action="store_true",
        help="Generate semi-realistic synthetic scores without loading or running any model.",
    )
    parser.add_argument(
        "--fake-seed",
        type=int,
        default=31,
        help="Seed for deterministic fake scoring.",
    )
    parser.add_argument(
        "--fake-script-margin",
        type=float,
        default=1.15,
        help="Typical Hant/Hans log-prob margin for same-plus-distinct mappings.",
    )
    parser.add_argument(
        "--fake-ambiguous-margin",
        type=float,
        default=0.55,
        help="Typical script margin for one-simplified-to-many-traditional mappings.",
    )
    parser.add_argument(
        "--fake-control-margin",
        type=float,
        default=1.75,
        help="Typical script margin for OpenCC one-to-one control mappings.",
    )
    parser.add_argument(
        "--fake-noise",
        type=float,
        default=0.75,
        help="Event-level Gaussian noise added to fake delta_logp.",
    )
    parser.add_argument(
        "--fake-pair-bias-sd",
        type=float,
        default=0.35,
        help="Stable pair-level Gaussian bias toward either script.",
    )
    parser.add_argument(
        "--fake-mixed-script-shrink",
        type=float,
        default=0.55,
        help="Multiplier applied to script margin for mixed-script lines.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip rows already present in --out-jsonl.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete --out-jsonl before scoring instead of resuming from it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.limit_events is not None and args.limit_events <= 0:
        raise ValueError("--limit-events must be positive when provided")
    if args.max_prefix_tokens is not None and args.max_prefix_tokens <= 0:
        raise ValueError("--max-prefix-tokens must be positive when provided")
    if not args.fake_scores and not args.model:
        raise ValueError("--model is required unless --fake-scores is used")
    if args.fake_noise < 0 or args.fake_pair_bias_sd < 0:
        raise ValueError("Fake noise settings must be non-negative")

    out_jsonl = ensure_under_output_root(Path(args.out_jsonl))
    events = load_jsonl(Path(args.events))
    if args.limit_events is not None:
        events = events[: args.limit_events]
    if not events:
        raise ValueError(f"No events to score in {args.events}")
    pending_events, skipped_completed = prepare_events_for_scoring(
        events,
        out_jsonl=out_jsonl,
        resume=not args.no_resume,
        overwrite=bool(args.overwrite_output),
    )
    if not pending_events:
        print(f"Output JSONL: {out_jsonl}")
        print(f"Events already scored: {skipped_completed}")
        print("No pending events to score.")
        return

    if args.fake_scores:
        model_name = args.model or "fake-script-preference-v1"
        written = fake_score_incremental(
            pending_events,
            args=args,
            model_name=model_name,
            out_jsonl=out_jsonl,
        )
        print(f"Model: {model_name}")
        print("Mode: fake scores (no model loaded)")
        print(f"Events already scored: {skipped_completed}")
        print(f"Events scored this run: {written}")
        print(f"Output JSONL: {out_jsonl}")
        return

    tokenizer, model, device = load_model_and_tokenizer(args)
    add_bos = should_add_bos(args.add_bos, tokenizer)

    iterator = range(0, len(pending_events), args.batch_size)
    if _TQDM:
        iterator = tqdm(
            iterator,
            total=(len(pending_events) + args.batch_size - 1) // args.batch_size,
            desc="score",
            unit="batch",
        )

    written = 0
    for start in iterator:
        batch_events = pending_events[start : start + args.batch_size]
        requests: list[CandidateRequest] = []
        event_errors: dict[int, str] = {}
        for offset, event in enumerate(batch_events):
            event_index = start + offset
            try:
                simplified = event["simplified"]
                traditional = event["traditional"]
                if not (isinstance(simplified, str) and isinstance(traditional, str)):
                    raise TypeError("simplified/traditional fields must be strings")
                requests.append(
                    build_candidate_request(
                        event_index=offset,
                        event=event,
                        candidate_name="simplified",
                        candidate_text=simplified,
                        tokenizer=tokenizer,
                        add_bos=add_bos,
                        max_prefix_tokens=args.max_prefix_tokens,
                    )
                )
                requests.append(
                    build_candidate_request(
                        event_index=offset,
                        event=event,
                        candidate_name="traditional",
                        candidate_text=traditional,
                        tokenizer=tokenizer,
                        add_bos=add_bos,
                        max_prefix_tokens=args.max_prefix_tokens,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                event_errors[offset] = str(exc)

        scores = score_candidate_requests(
            requests,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
        batch_scored = annotate_scores(batch_events, requests, scores, model_name=args.model)
        for offset, error in event_errors.items():
            batch_scored[offset]["model_name"] = args.model
            batch_scored[offset]["score_error"] = error
        append_jsonl(out_jsonl, batch_scored)
        written += len(batch_scored)

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Events already scored: {skipped_completed}")
    print(f"Events scored this run: {written}")
    print(f"Output JSONL: {out_jsonl}")


if __name__ == "__main__":
    main()
