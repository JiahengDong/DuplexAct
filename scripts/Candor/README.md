# Candor extractors

Candor audio cannot be redistributed. Download the official corpus (per-conversation
folders with audio + **Backbiter** transcripts), then run the four pipelines below.
These are the only extractors used to build the evaluation set.

Always pass `--candor_root` / `--results_dir` (or `CANDOR_ROOT`). Hardcoded defaults
are leftovers from the original cluster and will not exist on your machine.

GPT filtering reads `OPENAI_API_KEY` from the environment.

All four pipelines read `transcription/transcript_backbiter.csv`.
