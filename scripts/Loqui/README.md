# Loqui extractors

The 131 Loqui evaluation clips are on
[Hugging Face](https://huggingface.co/datasets/LincolnD/DuplexAct),
redistributed with permission of the Loqui authors. Attribute
Passonneau & Sachar (2014):
[doi:10.7916/D82R3PW9](https://doi.org/10.7916/D82R3PW9).

You do **not** need these scripts to run the benchmark. They are the
extractors used to build those clips from the official `LoquiPackage`
(`Transcriptions/*.final.trs` + `*.san.wav`).

Always pass `--loqui_root` / `--results_dir` (or `LOQUI_ROOT`). Hardcoded
defaults are leftovers from the original cluster and will not exist on your
machine. `--loqui_root` may be `LoquiPackage`, its parent, or the
`Transcriptions` folder.

GPT filtering reads `OPENAI_API_KEY` from the environment.
