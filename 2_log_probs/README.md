# 2_log_probs Experiment Setup

This directory sets up the next-token log-prob experiment for simplified vs. traditional character choices. The scripts here do not need to generate text: the scorer only measures the model probability assigned to two candidate continuations after the same prefix.

## Current Recommended Runbook

Use this section when you just want to run the experiment.

Create the full extracted event file with OpenCC one-to-one controls:

```bash
python 2_log_probs/extract_ambiguity_events.py \
  --include-controls \
  --per-pair-limit 100 \
  --out-jsonl 2_log_probs/out/ambiguity_events_with_controls.jsonl \
  --summary-out 2_log_probs/out/ambiguity_events_with_controls_summary.json
```

Create the 50k subset for the first real model pass:

```bash
python 2_log_probs/subset_events.py \
  --events 2_log_probs/out/ambiguity_events_with_controls.jsonl \
  --n 50000 \
  --seed 53 \
  --out-jsonl 2_log_probs/out/ambiguity_events_with_controls_50k.jsonl
```

Score the 50k subset with the small baseline model:

```bash
python 2_log_probs/score_next_token_logprobs.py \
  --events 2_log_probs/out/ambiguity_events_with_controls_50k.jsonl \
  --model Qwen/Qwen3-0.6B-Base \
  --device auto \
  --batch-size 8 \
  --max-prefix-tokens 1024 \
  --out-jsonl 2_log_probs/out/qwen3_0_6b_base_50k_scored_logprobs.jsonl
```

Scoring checkpoints automatically. If you stop it, rerun the same command and it skips already-scored `event_id`s. Check progress:

```bash
wc -l 2_log_probs/out/qwen3_0_6b_base_50k_scored_logprobs.jsonl
wc -l 2_log_probs/out/ambiguity_events_with_controls_50k.jsonl
```

Summarize and make plots:

```bash
python 2_log_probs/summarize_logprobs.py \
  --scored-jsonl 2_log_probs/out/qwen3_0_6b_base_50k_scored_logprobs.jsonl \
  --out-dir 2_log_probs/out/qwen3_0_6b_base_50k_summary
```

Inspect:

```text
2_log_probs/out/qwen3_0_6b_base_50k_summary/quick_report.md
2_log_probs/out/qwen3_0_6b_base_50k_summary/aggregate_summary.csv
2_log_probs/out/qwen3_0_6b_base_50k_summary/pair_summary.csv
2_log_probs/out/qwen3_0_6b_base_50k_summary/*.png
```

### Larger Model To Run Later

Yes, it makes sense to run a slightly larger model after the small baseline. The clean next step is `Qwen/Qwen2.5-1.5B`, because it is still manageable on a Mac, is a base causal LM, and should give a stronger comparison without jumping to an 8B run.

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B
```

Then:

```bash
python 2_log_probs/score_next_token_logprobs.py \
  --events 2_log_probs/out/ambiguity_events_with_controls_50k.jsonl \
  --model Qwen/Qwen2.5-1.5B \
  --device auto \
  --batch-size 4 \
  --max-prefix-tokens 1024 \
  --out-jsonl 2_log_probs/out/qwen2_5_1_5b_50k_scored_logprobs.jsonl
```

Use `--batch-size 1` if MPS memory is tight. Summarize it with:

```bash
python 2_log_probs/summarize_logprobs.py \
  --scored-jsonl 2_log_probs/out/qwen2_5_1_5b_50k_scored_logprobs.jsonl \
  --out-dir 2_log_probs/out/qwen2_5_1_5b_50k_summary
```

Core metric:

```text
delta_logp = log P(traditional_char | prefix) - log P(simplified_char | prefix)
```

Positive `delta_logp` means the model favored the traditional character at that point in context. Negative `delta_logp` means it favored the simplified character.

## 1. Extract Ambiguity Events

Default extraction uses:

- `../data/disambiguation-data/wiki_one_to_multi_tokenized_clean.json`
- `../data/zhwiki-clean/clean_zh_hant.txt`
- `../data/zhwiki-clean/clean_zh_hans.txt`

Run a small smoke extraction first:

```bash
python3 2_log_probs/extract_ambiguity_events.py \
  --max-lines 1000 \
  --per-pair-limit 2 \
  --out-jsonl 2_log_probs/out/smoke_events.jsonl \
  --summary-out 2_log_probs/out/smoke_events_summary.json
```

Run the main extraction:

```bash
python3 2_log_probs/extract_ambiguity_events.py \
  --per-pair-limit 100 \
  --out-jsonl 2_log_probs/out/ambiguity_events.jsonl \
  --summary-out 2_log_probs/out/ambiguity_events_summary.json
```

To include OpenCC one-to-one controls:

```bash
python3 2_log_probs/extract_ambiguity_events.py \
  --include-controls \
  --per-pair-limit 100 \
  --out-jsonl 2_log_probs/out/ambiguity_events_with_controls.jsonl \
  --summary-out 2_log_probs/out/ambiguity_events_with_controls_summary.json
```

Events are reservoir-sampled per `(context_script, simplified, traditional)` so full-corpus scans stay bounded.

If the extracted file is too large for a first model pass, make a reproducible total-size subset:

```bash
python3 2_log_probs/subset_events.py \
  --events 2_log_probs/out/ambiguity_events_with_controls.jsonl \
  --n 50000 \
  --seed 53 \
  --out-jsonl 2_log_probs/out/ambiguity_events_with_controls_50k.jsonl
```

## 2. Dry-Run With Fake Scores

Before running an LLM, generate semi-realistic fake scores from extracted events. This uses the same scored JSONL schema as the real scorer, so the summary and visualization outputs are identical in shape.

```bash
python3 2_log_probs/score_next_token_logprobs.py \
  --fake-scores \
  --events 2_log_probs/out/smoke_events.jsonl \
  --out-jsonl 2_log_probs/out/fake_scored_logprobs.jsonl
```

The fake scorer is deterministic by default. It makes Hant contexts tend to favor traditional characters, Hans contexts tend to favor simplified characters, ambiguous one-to-many mappings noisier, OpenCC one-to-one controls easier, and mixed-script lines less decisive. Tune it if you want to stress-test the plots:

```bash
python3 2_log_probs/score_next_token_logprobs.py \
  --fake-scores \
  --events 2_log_probs/out/smoke_events.jsonl \
  --fake-seed 99 \
  --fake-ambiguous-margin 0.25 \
  --fake-noise 1.25 \
  --out-jsonl 2_log_probs/out/fake_scored_noisy.jsonl
```

Then summarize and visualize:

```bash
python3 2_log_probs/summarize_logprobs.py \
  --scored-jsonl 2_log_probs/out/fake_scored_logprobs.jsonl \
  --out-dir 2_log_probs/out/fake_summary
```

This is the main pre-LLM check. Inspect `2_log_probs/out/fake_summary/quick_report.md`, the CSVs, and the PNGs before spending time on model scoring.

## 3. Score Candidate Log Probs With an LLM

You run this step with the causal LM you want to test. The setup supports Hugging Face model names or local model paths. It does not generate continuations; it scores only the simplified and traditional candidate characters.

Example with a local or cached model:

```bash
python3 2_log_probs/score_next_token_logprobs.py \
  --events 2_log_probs/out/ambiguity_events_with_controls_50k.jsonl \
  --model /path/to/local/causal-lm \
  --local-files-only \
  --device auto \
  --batch-size 8 \
  --out-jsonl 2_log_probs/out/scored_logprobs.jsonl
```

If the model needs remote loading, omit `--local-files-only`. If prefixes are too long for a model, add a left-truncation cap:

```bash
python3 2_log_probs/score_next_token_logprobs.py \
  --events 2_log_probs/out/ambiguity_events_with_controls_50k.jsonl \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --batch-size 8 \
  --max-prefix-tokens 1024 \
  --out-jsonl 2_log_probs/out/scored_logprobs.jsonl
```

The scored JSONL records:

- `simplified_logprob`
- `traditional_logprob`
- `delta_logp`
- candidate token IDs and token strings
- `single_token_pair`, for the strict next-token subset where both candidates are one token
- tokenization mode fields showing whether candidate IDs were derived as a direct prefix delta

Scoring is resumable by default. Each completed batch is appended to `--out-jsonl`; if a run is interrupted, rerun the same command and existing `event_id`s in that file will be skipped. Use `--overwrite-output` only when you intentionally want to restart from scratch.

## 4. Summarize Results

```bash
python3 2_log_probs/summarize_logprobs.py \
  --scored-jsonl 2_log_probs/out/scored_logprobs.jsonl \
  --out-dir 2_log_probs/out/summary
```

This writes:

- `aggregate_summary.csv`: grouped by context, mapping source/kind, and `single_token_pair`
- `pair_summary.csv`: per simplified/traditional pair
- `strongest_examples.csv`: largest absolute `delta_logp` examples for inspection
- `summary_report.json`: row counts and skipped-row counts
- `quick_report.md`: compact human-readable summary
- `delta_distribution.png`: distribution of event-level deltas by context
- `aggregate_mean_delta.png`: group-level mean delta bars
- `top_pair_mean_delta.png`: strongest pair-level mean deltas
- `single_token_share.png`: single-token vs multi-token pair counts

## Non-LLM Checks

These commands do not run any model:

```bash
python3 - <<'PY'
from pathlib import Path

for path in [
    Path("2_log_probs/extract_ambiguity_events.py"),
    Path("2_log_probs/score_next_token_logprobs.py"),
    Path("2_log_probs/summarize_logprobs.py"),
]:
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(f"ok: {path}")
PY

python3 2_log_probs/extract_ambiguity_events.py --help
python3 2_log_probs/score_next_token_logprobs.py --help
python3 2_log_probs/summarize_logprobs.py --help

python3 2_log_probs/summarize_logprobs.py \
  --scored-jsonl 2_log_probs/test_fixtures/scored_events_tiny.jsonl \
  --out-dir 2_log_probs/out/test_summary

python3 2_log_probs/score_next_token_logprobs.py \
  --fake-scores \
  --events 2_log_probs/out/smoke_events.jsonl \
  --out-jsonl 2_log_probs/out/fake_scored_logprobs.jsonl

python3 2_log_probs/summarize_logprobs.py \
  --scored-jsonl 2_log_probs/out/fake_scored_logprobs.jsonl \
  --out-dir 2_log_probs/out/fake_summary
```

All script-generated outputs are constrained to `2_log_probs/out/`.
