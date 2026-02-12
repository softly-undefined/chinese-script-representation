from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_path(root: Path) -> Path:
    candidates = [
        root / "data" / "disambiguation" / "wiki_one_to_multi.json",
        root / "data" / "disambiguation-data" / "wiki_one_to_multi.json",
        root / "data" / "disambiguation-data" / "wiki_one_to_multi_tokenized_clean.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def pick_device(requested: str) -> torch.device:
    req = requested.strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected list JSON in {path}, got {type(data).__name__}")
    return data


def extract_one_to_two_mappings(records: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """
    Keep only mappings where:
    - one simplified char maps to exactly two traditional chars
    - both traditional chars are distinct
    - neither traditional char equals simplified
    """
    mappings: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    iterator = records
    if _TQDM:
        iterator = tqdm(records, desc="filter_mappings", unit="item")

    for item in iterator:
        if not isinstance(item, dict):
            continue
        s = item.get("simplified")
        trad_list = item.get("traditional", []) or []
        if not (isinstance(s, str) and len(s) == 1):
            continue
        if not (isinstance(trad_list, list) and len(trad_list) == 2):
            continue

        t_chars = [t for t in trad_list if isinstance(t, str) and len(t) == 1]
        if len(t_chars) != 2:
            continue

        t1, t2 = t_chars[0], t_chars[1]
        if t1 == t2:
            continue
        if t1 == s or t2 == s:
            continue

        # Canonical order for stable dedupe.
        a, b = sorted([t1, t2])
        key = (s, a, b)
        if key in seen:
            continue
        seen.add(key)
        mappings.append(key)

    if not mappings:
        raise ValueError(
            "No valid one-to-two mappings found after filtering "
            "(need exactly two traditional chars, both different from simplified)."
        )
    return mappings


def char_layer_vectors_and_tokens(
    char: str, tokenizer, model, device: torch.device
) -> tuple[torch.Tensor, list[str], list[int]]:
    encoded = tokenizer(char, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    content_tokens = all_tokens[1:-1]
    content_ids = input_ids[1:-1]

    seq_len = hidden_states[0].shape[1]
    if seq_len < 3:
        raise RuntimeError(f"Unexpected sequence length for char '{char}': {seq_len}")

    per_layer = []
    for h in hidden_states:
        vec = h[0, 1 : seq_len - 1, :].mean(dim=0).detach().cpu()
        per_layer.append(vec)
    return torch.stack(per_layer), content_tokens, content_ids


def build_char_cache(chars: list[str], tokenizer, model, device: torch.device):
    cache: dict[str, dict[str, Any]] = {}
    iterator = chars
    if _TQDM:
        iterator = tqdm(chars, desc="embed_chars", unit="char")
    for ch in iterator:
        vecs, tokens, ids = char_layer_vectors_and_tokens(ch, tokenizer, model, device)
        cache[ch] = {"vectors": vecs, "tokens": tokens, "token_ids": ids}
    return cache


def main() -> None:
    root = repo_root()
    default_out = Path(__file__).resolve().parent / "one_to_two_layerwise_cag.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Compute layer-wise CAG for one simplified -> two traditional mappings:\n"
            "CAG = |cos(s,t1) - cos(s,t2)|"
        )
    )
    parser.add_argument(
        "--mappings-json",
        default=str(default_data_path(root)),
        help="Path to wiki_one_to_multi JSON",
    )
    parser.add_argument(
        "--model",
        default="bert-base-multilingual-cased",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto/cpu/cuda/mps",
    )
    parser.add_argument(
        "--out-csv",
        default=str(default_out),
        help="Output CSV path for layer-wise CAG mappings",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer from local cache only (no network calls).",
    )
    args = parser.parse_args()

    mapping_path = Path(args.mappings_json)
    records = load_records(mapping_path)
    mappings = extract_one_to_two_mappings(records)
    chars = sorted({c for s, t1, t2 in mappings for c in (s, t1, t2)})

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=bool(args.local_files_only),
    )
    model = AutoModel.from_pretrained(
        args.model,
        local_files_only=bool(args.local_files_only),
    ).to(device)
    model.eval()

    cache = build_char_cache(chars, tokenizer, model, device)
    num_layers = int(next(iter(cache.values()))["vectors"].shape[0])
    layer_names = ["embedding"] + [f"layer_{i}" for i in range(1, num_layers)]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "mapping_index",
        "simplified",
        "traditional_1",
        "traditional_2",
        "layer_index",
        "layer_name",
        "cosine_s_t1",
        "cosine_s_t2",
        "cosine_t1_t2",
        "cag",
        "simplified_tokens",
        "simplified_token_ids",
        "traditional_1_tokens",
        "traditional_1_token_ids",
        "traditional_2_tokens",
        "traditional_2_token_ids",
    ]

    iterator = mappings
    if _TQDM:
        iterator = tqdm(mappings, desc="compute_cag", unit="mapping")

    row_count = 0
    with io.open(out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for i, (s, t1, t2) in enumerate(iterator):
            s_vecs = cache[s]["vectors"]
            t1_vecs = cache[t1]["vectors"]
            t2_vecs = cache[t2]["vectors"]

            cos_s_t1 = F.cosine_similarity(s_vecs, t1_vecs, dim=1)
            cos_s_t2 = F.cosine_similarity(s_vecs, t2_vecs, dim=1)
            cos_t1_t2 = F.cosine_similarity(t1_vecs, t2_vecs, dim=1)
            cag = torch.abs(cos_s_t1 - cos_s_t2)

            for layer_idx, layer_name in enumerate(layer_names):
                writer.writerow(
                    {
                        "mapping_index": i,
                        "simplified": s,
                        "traditional_1": t1,
                        "traditional_2": t2,
                        "layer_index": layer_idx,
                        "layer_name": layer_name,
                        "cosine_s_t1": float(cos_s_t1[layer_idx].item()),
                        "cosine_s_t2": float(cos_s_t2[layer_idx].item()),
                        "cosine_t1_t2": float(cos_t1_t2[layer_idx].item()),
                        "cag": float(cag[layer_idx].item()),
                        "simplified_tokens": json.dumps(cache[s]["tokens"], ensure_ascii=False),
                        "simplified_token_ids": json.dumps(cache[s]["token_ids"], ensure_ascii=False),
                        "traditional_1_tokens": json.dumps(cache[t1]["tokens"], ensure_ascii=False),
                        "traditional_1_token_ids": json.dumps(cache[t1]["token_ids"], ensure_ascii=False),
                        "traditional_2_tokens": json.dumps(cache[t2]["tokens"], ensure_ascii=False),
                        "traditional_2_token_ids": json.dumps(cache[t2]["token_ids"], ensure_ascii=False),
                    }
                )
                row_count += 1

    print(f"Mappings JSON: {mapping_path}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Mappings processed: {len(mappings)}")
    print(f"Layers per mapping: {num_layers}")
    print(f"Rows written: {row_count}")
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    main()
