# This file should operate similarly to ../mBERT_test.py, but over the entire data/disambiguation-data/opencc_one_to_one_pairs_tokenized_clean.json
# it should save the mappings for each layer to a csv in this directory

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


def default_pairs_path(root: Path) -> Path:
    preferred = root / "data" / "disambiguation-data" / "opencc_one_to_one_pairs_tokenized_clean.json"
    if preferred.exists():
        return preferred
    fallback = root / "discovering_traditional" / "opencc_one_to_one_pairs_tokenized_clean.json"
    return fallback


def pick_device(requested: str) -> torch.device:
    req = requested.strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def load_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_pairs = payload.get("pairs", []) if isinstance(payload, dict) else []
    if not isinstance(raw_pairs, list):
        raise TypeError(f"`pairs` must be a list in {path}")

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        trad = item.get("traditional")
        simp = item.get("simplified")
        if not (isinstance(trad, str) and isinstance(simp, str) and len(trad) == 1 and len(simp) == 1):
            continue
        pair = (trad, simp)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    if not pairs:
        raise ValueError(f"No valid one-to-one pairs found in {path}")
    return pairs


def char_layer_vectors_and_tokens(
    char: str, tokenizer, model, device: torch.device
) -> tuple[torch.Tensor, list[str], list[int]]:
    """
    Returns:
      - layer vectors: [num_layers_plus_embedding, hidden_size]
      - content tokens for the char
      - content token ids for the char
    """
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
    default_out = Path(__file__).resolve().parent / "one_to_one_layerwise_distances.csv"

    parser = argparse.ArgumentParser(
        description="Compute mBERT layer-wise distances for all cleaned one-to-one pairs."
    )
    parser.add_argument(
        "--pairs-json",
        default=str(default_pairs_path(root)),
        help="Path to opencc_one_to_one_pairs_tokenized_clean.json",
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
        help="Output CSV path for layer-wise distance mappings",
    )
    args = parser.parse_args()

    pairs = load_pairs(Path(args.pairs_json))
    chars = sorted({c for pair in pairs for c in pair})

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    cache = build_char_cache(chars, tokenizer, model, device)

    num_layers = int(next(iter(cache.values()))["vectors"].shape[0])
    layer_names = ["embedding"] + [f"layer_{i}" for i in range(1, num_layers)]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "pair_index",
        "traditional",
        "simplified",
        "layer_index",
        "layer_name",
        "l2_distance",
        "cosine_distance",
        "cosine_similarity",
        "traditional_tokens",
        "traditional_token_ids",
        "simplified_tokens",
        "simplified_token_ids",
    ]

    pair_iter = pairs
    if _TQDM:
        pair_iter = tqdm(pairs, desc="pair_distances", unit="pair")

    row_count = 0
    with io.open(out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for i, (traditional, simplified) in enumerate(pair_iter):
            trad = cache[traditional]
            simp = cache[simplified]

            trad_vecs = trad["vectors"]
            simp_vecs = simp["vectors"]

            l2 = torch.norm(simp_vecs - trad_vecs, p=2, dim=1)
            cos_sim = F.cosine_similarity(simp_vecs, trad_vecs, dim=1)
            cos_dist = 1.0 - cos_sim

            for layer_idx, layer_name in enumerate(layer_names):
                writer.writerow(
                    {
                        "pair_index": i,
                        "traditional": traditional,
                        "simplified": simplified,
                        "layer_index": layer_idx,
                        "layer_name": layer_name,
                        "l2_distance": float(l2[layer_idx].item()),
                        "cosine_distance": float(cos_dist[layer_idx].item()),
                        "cosine_similarity": float(cos_sim[layer_idx].item()),
                        "traditional_tokens": json.dumps(trad["tokens"], ensure_ascii=False),
                        "traditional_token_ids": json.dumps(trad["token_ids"], ensure_ascii=False),
                        "simplified_tokens": json.dumps(simp["tokens"], ensure_ascii=False),
                        "simplified_token_ids": json.dumps(simp["token_ids"], ensure_ascii=False),
                    }
                )
                row_count += 1

    print(f"Pairs JSON: {args.pairs_json}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Pairs processed: {len(pairs)}")
    print(f"Layers per pair: {num_layers}")
    print(f"Rows written: {row_count}")
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    main()
