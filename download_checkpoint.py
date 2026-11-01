"""Download and pin an externally trained SAC checkpoint from Hugging Face.

Does not train or redistribute the checkpoint in this repository.
Always record the exact remote revision and local SHA-256 for provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ID = "hhmm1122/fetch-pickandplace-sac-her"
FILENAME = "best_model.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("upstream-model"))

    from huggingface_hub import hf_hub_download

    # Use the immutable revision that was evaluated and recorded in the evidence.
    # Override only when deliberately testing a different upstream checkpoint.
    parser.add_argument("--revision", default="c817f4a3033fb4bde985bfcfe93ed5d8b949b04b")
    args = parser.parse_args()
    remote_sha = args.revision
    if not remote_sha or len(remote_sha) != 40:
        raise ValueError("revision must be a full 40-character Hugging Face commit SHA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        revision=remote_sha,
        local_dir=str(args.output_dir),
    ))
    manifest = {
        "model_id": REPO_ID,
        "filename": FILENAME,
        "revision": remote_sha,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_path": str(checkpoint.resolve()),
        "source": "https://huggingface.co/" + REPO_ID,
        "license_note": "Check upstream terms; no redistribution or license claim is made here.",
    }
    dest = args.output_dir / "model_provenance.json"
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
