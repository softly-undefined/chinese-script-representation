from __future__ import annotations

import argparse
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
    return Path(__file__).resolve().parents[1]


def pick_device(requested: str) -> torch.device:
    req = requested.strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def load_base_data(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"base_data.json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected list JSON at {path}, got {type(data).__name__}")
    return data


def collect_unique_chars(records: list[dict[str, Any]]) -> list[str]:
    chars: set[str] = set()
    for item in records:
        simplified = item.get("simplified")
        if isinstance(simplified, str) and simplified:
            chars.add(simplified)
        for trad in item.get("traditional", []) or []:
            if isinstance(trad, str) and trad:
                chars.add(trad)
    return sorted(chars)


def collect_pairs(records: list[dict[str, Any]], include_identical: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in records:
        simplified = item.get("simplified")
        if not isinstance(simplified, str) or not simplified:
            continue
        for trad in item.get("traditional", []) or []:
            if not isinstance(trad, str) or not trad:
                continue
            if not include_identical and trad == simplified:
                continue
            pair = (simplified, trad)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def char_layer_vectors(char: str, tokenizer, model, device: torch.device) -> torch.Tensor:
    """
    Return [num_layers_plus_embedding, hidden_size] for one character.
    If tokenized into multiple wordpieces, average those subtoken vectors per layer.
    """
    encoded = tokenizer(char, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    # hidden_states is tuple length (num_layers + 1), each [1, seq_len, hidden]
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")

    seq_len = hidden_states[0].shape[1]
    if seq_len < 3:
        raise RuntimeError(f"Unexpected tokenized length for char '{char}': {seq_len}")

    # Positions 1:-1 correspond to content tokens (exclude [CLS], [SEP]).
    start = 1
    end = seq_len - 1

    per_layer: list[torch.Tensor] = []
    for h in hidden_states:
        vec = h[0, start:end, :].mean(dim=0).detach().cpu()
        per_layer.append(vec)
    return torch.stack(per_layer)  # [L+1, H]


def build_char_vector_cache(
    chars: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    iterator = chars
    if _TQDM:
        iterator = tqdm(chars, desc="embed_chars", unit="char")
    for ch in iterator:
        cache[ch] = char_layer_vectors(ch, tokenizer, model, device)
    return cache


def layerwise_distances(v1: torch.Tensor, v2: torch.Tensor) -> dict[str, list[float]]:
    """
    v1, v2: [L+1, H]
    """
    l2 = torch.norm(v1 - v2, p=2, dim=1)
    cos_sim = F.cosine_similarity(v1, v2, dim=1)
    cos_dist = 1.0 - cos_sim
    return {
        "l2_distance": [float(x) for x in l2.tolist()],
        "cosine_distance": [float(x) for x in cos_dist.tolist()],
        "cosine_similarity": [float(x) for x in cos_sim.tolist()],
    }


def write_outputs(
    out_pt: Path,
    out_json: Path,
    model_name: str,
    device: torch.device,
    chars: list[str],
    char_cache: dict[str, torch.Tensor],
    pairs: list[tuple[str, str]],
) -> None:
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    matrix = torch.stack([char_cache[ch] for ch in chars])  # [N, L+1, H]
    num_layers_plus_embedding = int(matrix.shape[1])
    hidden_size = int(matrix.shape[2])
    layer_names = ["embedding"] + [f"layer_{i}" for i in range(1, num_layers_plus_embedding)]

    pair_results: list[dict[str, Any]] = []
    for simplified, traditional in pairs:
        d = layerwise_distances(char_cache[simplified], char_cache[traditional])
        pair_results.append(
            {
                "simplified": simplified,
                "traditional": traditional,
                "layer_names": layer_names,
                "l2_distance": d["l2_distance"],
                "cosine_distance": d["cosine_distance"],
                "cosine_similarity": d["cosine_similarity"],
            }
        )

    torch_payload = {
        "model_name": model_name,
        "device_used_for_load": str(device),
        "chars": chars,
        "layer_names": layer_names,
        "vectors": matrix,  # [num_chars, num_layers_plus_embedding, hidden_size]
        "pairs": pairs,
    }
    torch.save(torch_payload, out_pt)

    json_payload = {
        "model_name": model_name,
        "num_chars": len(chars),
        "num_pairs": len(pairs),
        "num_layers_plus_embedding": num_layers_plus_embedding,
        "hidden_size": hidden_size,
        "layer_names": layer_names,
        "pair_distances": pair_results,
        "tensor_file": str(out_pt),
    }
    with io.open(out_json, "w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Compute layer-wise geometric distances between simplified characters and "
            "traditional counterparts using mBERT representations."
        )
    )
    parser.add_argument(
        "--base-json",
        default=str(root / "data" / "disambiguation-data" / "base_data.json"),
        help="Path to base_data.json",
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
        "--include-identical",
        action="store_true",
        help="Include identical simplified==traditional pairs",
    )
    parser.add_argument(
        "--out-pt",
        default=str(root / "data" / "disambiguation-data" / "mbert_layerwise_vectors.pt"),
        help="Output .pt for all character layer-wise vectors",
    )
    parser.add_argument(
        "--out-json",
        default=str(root / "data" / "disambiguation-data" / "mbert_layerwise_pair_distances.json"),
        help="Output .json for layer-wise pair distances",
    )
    args = parser.parse_args()

    base_json = Path(args.base_json)
    out_pt = Path(args.out_pt)
    out_json = Path(args.out_json)
    device = pick_device(args.device)

    records = load_base_data(base_json)
    chars = collect_unique_chars(records)
    pairs = collect_pairs(records, include_identical=args.include_identical)
    if not chars:
        raise ValueError(f"No characters found in {base_json}")
    if not pairs:
        raise ValueError("No simplified/traditional pairs found in base_data.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    char_cache = build_char_vector_cache(chars, tokenizer, model, device)
    write_outputs(
        out_pt=out_pt,
        out_json=out_json,
        model_name=args.model,
        device=device,
        chars=chars,
        char_cache=char_cache,
        pairs=pairs,
    )

    sample_pair = pairs[0]
    sample = layerwise_distances(char_cache[sample_pair[0]], char_cache[sample_pair[1]])
    print(f"Base JSON: {base_json}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Characters embedded: {len(chars)}")
    print(f"Pairs analyzed: {len(pairs)}")
    print(f"Saved vectors: {out_pt}")
    print(f"Saved pair distances: {out_json}")
    print(
        "Sample pair first/last layer L2: "
        f"{sample['l2_distance'][0]:.6f} -> {sample['l2_distance'][-1]:.6f}"
    )


if __name__ == "__main__":
    main()
