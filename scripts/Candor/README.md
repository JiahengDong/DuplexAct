# Candor extractors

Candor audio cannot be redistributed. Download the official corpus, then run the **relaxed** pipelines documented in the top-level README.

Always pass `--candor_root` / `--results_dir` (or `CANDOR_ROOT`). Hardcoded defaults are leftovers from the original cluster and will not exist on your machine.

GPT filtering reads `OPENAI_API_KEY` from the environment (no keys are stored in these files).

Older `*_backbiter.py` / `extract_interruptions.py` (non-relaxed) files are kept for provenance. The evaluation set used the `*_relaxed.py` extractors plus `extract_turn_taking_backbiter.py` and `extract_pause_candidates.py`.
