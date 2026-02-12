# intro to mBERT. use one example sentence to test that tokenization and processing is working

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "bert-base-multilingual-cased"
SIMPLIFIED_CHAR = "云"
TRADITIONAL_CHAR = "雲"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def char_layer_vectors(char: str, tokenizer, model, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(char, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    # [CLS] token is index 0, [SEP] is last index. Content is 1:-1.
    seq_len = hidden_states[0].shape[1]
    if seq_len < 3:
        raise RuntimeError(f"Unexpected sequence length for char '{char}': {seq_len}")

    per_layer = []
    for h in hidden_states:
        vec = h[0, 1 : seq_len - 1, :].mean(dim=0).detach().cpu()
        per_layer.append(vec)
    return torch.stack(per_layer)  # [L+1, H]


def main() -> None:
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    simp_vecs = char_layer_vectors(SIMPLIFIED_CHAR, tokenizer, model, device)
    trad_vecs = char_layer_vectors(TRADITIONAL_CHAR, tokenizer, model, device)

    l2 = torch.norm(simp_vecs - trad_vecs, p=2, dim=1)
    cos_sim = F.cosine_similarity(simp_vecs, trad_vecs, dim=1)
    cos_dist = 1.0 - cos_sim

    layer_names = ["embedding"] + [f"layer_{i}" for i in range(1, simp_vecs.shape[0])]

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Pair: {SIMPLIFIED_CHAR} (simplified) vs {TRADITIONAL_CHAR} (traditional)")
    print("")
    print("Layer-by-layer distances")
    print("layer\tl2_distance\tcosine_distance\tcosine_similarity")
    for idx, name in enumerate(layer_names):
        print(
            f"{name}\t{l2[idx].item():.6f}\t{cos_dist[idx].item():.6f}\t{cos_sim[idx].item():.6f}"
        )

    print("")
    print(
        "Trend summary (embedding -> last): "
        f"L2 {l2[0].item():.6f} -> {l2[-1].item():.6f}, "
        f"cos_dist {cos_dist[0].item():.6f} -> {cos_dist[-1].item():.6f}"
    )


if __name__ == "__main__":
    main()
