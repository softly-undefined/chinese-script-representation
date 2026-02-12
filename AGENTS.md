## Local Agent Rules

- Do **not** delete or overwrite any existing user comments at the top of files. If you need to add your own header comment, place it *after* the user’s comments.
- When adding long-running scripts or data processing, prefer showing a tqdm progress bar (if feasible) so progress is visible. Use the optional dependency approach: try/except import tqdm; fall back silently if unavailable.
