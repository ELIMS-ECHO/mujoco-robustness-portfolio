"""从 Hugging Face 下载并固定第三方训练的 SAC 模型检查点。

本脚本不会训练模型，也不会在本仓库中重新分发第三方权重。
必须记录准确的上游版本标识和本地 SHA-256，以便追溯模型来源。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ID = "hhmm1122/fetch-pickandplace-sac-her"
FILENAME = "best_model.zip"
DEFAULT_REVISION = "c817f4a3033fb4bde985bfcfe93ed5d8b949b04b"
EXPECTED_SHA256 = "1aad52960bc0b7baaa5249f3aa1a2de09009c300dd2b0bf61c9c51d7467b017b"


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

    # 使用已经完成评估并记录在证据文件中的不可变版本。
    # 只有在明确测试另一个上游模型版本时，才覆盖此参数。
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    remote_sha = args.revision
    if not remote_sha or len(remote_sha) != 40:
        raise ValueError("revision 必须是完整的 40 位 Hugging Face commit SHA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        revision=remote_sha,
        local_dir=str(args.output_dir),
    ))
    checkpoint_sha = sha256(checkpoint)
    if remote_sha == DEFAULT_REVISION and checkpoint_sha != EXPECTED_SHA256:
        raise RuntimeError(
            "下载的模型权重哈希与已核验的作品集基线不一致："
            f"expected {EXPECTED_SHA256}, got {checkpoint_sha}"
        )
    manifest = {
        "model_id": REPO_ID,
        "filename": FILENAME,
        "revision": remote_sha,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_path": str(checkpoint.resolve()),
        "source": "https://huggingface.co/" + REPO_ID,
        "license_note": "请核实上游授权条款；本仓库不声明模型权重许可，也不重新分发模型权重。",
    }
    dest = args.output_dir / "model_provenance.json"
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
