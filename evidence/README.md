# Public experiment evidence

This folder contains the machine-readable evidence for the 30-seed evaluation reported in [EVAL30_REPORT_2026-11-01.md](../EVAL30_REPORT_2026-11-01.md).

## Files

- `summary.csv`: compact, GitHub-previewable seven-condition summary.
- `summary.json`: full condition-level summary exported by the evaluation script.
- `episodes.csv.gz`: gzip-compressed original 210-row episode-level CSV. It is compressed only to keep the Git repository small; it is not a resampled or transformed dataset.
- `model_provenance.json`: upstream model identifier, immutable Hugging Face revision and checkpoint SHA-256.

## Reconstruct the raw CSV

Python 3.11:

```powershell
python -c "import gzip, pathlib; p=pathlib.Path('episodes.csv.gz'); pathlib.Path('episodes.csv').write_bytes(gzip.decompress(p.read_bytes()))"
```

Verify the reconstructed file against the SHA-256 below:

`4adea82b096b3f100a4d830b084a3bd46861ccd69c280d3e693a5c59a38c0dd1`

The compressed archive SHA-256 is:

`e88cf4890a7eaaca2ce43698e53ea8da4ced5a25867ec87cbcabe9ee0b542f32`

## Provenance and limitations

The 210 episodes were produced by a completed MuJoCo evaluation run on 2026-11-01, using one third-party SAC+HER checkpoint. The public repository preserves the original episode-level CSV bytes, aggregate summary and model provenance. The model weights are not included. The upstream model card does not clearly state a license; verify upstream terms before reuse.

The 30-seed evaluation is simulation evidence for one fixed checkpoint. It does not establish cross-checkpoint generalization or real-robot performance. The heuristic failure buckets are diagnostics, not validated physical labels.
