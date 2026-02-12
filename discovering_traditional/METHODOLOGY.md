# Methodology: Discovering Traditional Characters with OpenCC

## Goal
Produce two artifacts using OpenCC conversion data:

1. A list of strict `traditional:simplified` 1:1 character pairs.
2. A list of traditional-exclusive characters.

## Script
`discovering_traditional/extract_opencc_data.py`

## Data Source Strategy
The extractor uses OpenCC in two stages:

1. Preferred path (`config_dict_parse`):
   - Locate OpenCC config JSON files (`t2s.json`, `tw2s.json`, `twp2s.json`, `hk2s.json`).
   - Read dictionary file references from those configs.
   - Parse `.txt` dictionary entries into character-level mapping pairs.

2. Fallback path (`unicode_scan_via_opencc`):
   - If parseable dictionary files are not available (common in some installs), initialize OpenCC directly.
   - Scan Unicode CJK codepoint ranges.
   - Convert each character with OpenCC `t2s`.
   - Keep single-character conversion results as `(traditional, simplified)` candidate pairs.

This guarantees extraction even when OpenCC package internals differ across environments.

## Pair Construction Rules
From candidate `(trad, simp)` mappings:

- Build `trad -> {simp...}` and `simp -> {trad...}` maps.
- Keep a pair only if both are unique:
  - `len(trad_to_simp[trad]) == 1`
  - `len(simp_to_trad[simp]) == 1`
- By default, identity mappings (`trad == simp`) are excluded from the 1:1 output.
  - Use `--include-identical` to keep them.

## Traditional-Exclusive Character Rules
Define:
- `T = {all chars seen on traditional side}`
- `S = {all chars seen on simplified side}`

Traditional-exclusive list is:
- `T - S`
- with an extra guard to avoid pure identity-only artifacts.

## Outputs
- `discovering_traditional/opencc_one_to_one_pairs.json`
- `discovering_traditional/opencc_traditional_exclusive_chars.json`
- `discovering_traditional/opencc_traditional_exclusive_chars.txt`

Each JSON output includes:
- `method`: extraction method used (`config_dict_parse` or `unicode_scan_via_opencc`)
- `sources`: dictionary files used when available

## Observed Result Caveat
In this run, method was `unicode_scan_via_opencc`.
This produced a large traditional-exclusive set, but a very small strict 1:1 bijective set.

Interpretation:
- OpenCC contains many one-to-many / many-to-one relationships.
- Strict bijection filtering is intentionally conservative.
- If a larger “supplemental pairs” list is desired, relax the 1:1 requirement (e.g., keep one-to-one in `trad -> simp` only).
