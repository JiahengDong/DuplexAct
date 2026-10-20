# Candor instances

Candor audio cannot be redistributed, so the 1,235 Candor test instances are shipped as a fixed recipe instead of audio.

**Getting Candor.** Apply for access on the [BetterUp data request site](https://betterup-data-requests.herokuapp.com/). Once approved you receive download links (live for one week). Candor is also on [TalkBank](https://talkbank.org/ca/access/CANDOR.html): first get approval from the BetterUp site, then register at TalkBank (email address and password), then email macw@cmu.edu to request TalkBank CANDOR access. Cite Reece et al. (2023), *Science Advances* 9(13), eadf3197.

`rebuild_candor.py` reads the BetterUp release layout (`<conversation_id>/processed/<conversation_id>.mp3` and `processed/channel_map.json`) and was verified on it. The TalkBank copy is reorganised into 16 segments, so it may need re-arranging into that layout first.

```bash
# 1. Hub instances (AMI / CORAAL / Loqui / synthesized)
python scripts/unpack_hf_dataset.py --repo_id LincolnD/DuplexAct --out_dir evaluation/data

# 2. Candor instances, merged into the same folders and manifests
python scripts/Candor/rebuild_candor.py --candor_root /path/to/Candor --out_dir evaluation/data
```

The script cuts every wav out of the conversation audio, writes the original `metadata.json`, and regenerates each `<action>/manifest.json` from the folders on disk. It ends by checking every wav against a SHA-256 of its samples.

| Action | Candor instances | Files per instance |
|---|---:|---|
| Wait | 230 | `input.wav` |
| Backchannel | 106 | `input.wav` |
| Take turn | 155 | `input.wav` |
| Stop | 158 | `lead_in.wav`, `cue.wav` |
| Hold floor | 162 | `lead_in.wav`, `cue.wav` |
| Recover | 424 | `lead_in.wav`, `cue.wav` |

## Recipe format

`candor_instances.jsonl` has one instance per line:

- `action`, `sample_dir`, `conversation_id`
- `metadata`: the instance's `metadata.json`
- `files`: for each wav
  - `channel`: `L` or `R` of the stereo conversation mp3 (`channel_map.json` gives the participant), or `null` for an all-silent file
  - `start`, `length`: sample range at 16 kHz
  - `zero`: `[start, end)` sample ranges, relative to `start`, that are silenced (the other speaker's words)
  - `pad`: trailing silence in samples
  - `sha256`: SHA-256 of the int16 little-endian samples

Audio is decoded with `librosa.load(path, sr=16000, mono=False)` and written as 16-bit PCM. With librosa 0.11.0, soxr 1.0.0 and soundfile 0.13.1 (libsndfile 1.2.2) the output is bit-identical to the evaluated files. A different mp3 decoder or resampler can change individual samples; the script reports any wav whose checksum differs.
