# Loqui extractors

Loqui audio is not redistributed with this benchmark. Download the official
deposit yourself ([doi:10.7916/D82R3PW9](https://doi.org/10.7916/D82R3PW9);
`LoquiPackage` with `Transcriptions/*.final.trs` and `*.san.wav`), then run
the four pipelines below. These are the only Loqui extractors used to build
the evaluation set.

Always pass `--loqui_root` / `--results_dir` (or `LOQUI_ROOT`). Hardcoded
defaults are leftovers from the original cluster and will not exist on your
machine. `--loqui_root` may be `LoquiPackage`, its parent, or the
`Transcriptions` folder.

GPT filtering reads `OPENAI_API_KEY` from the environment.
