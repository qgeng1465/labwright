#!/bin/sh
# Publish the SciRecipe audit dataset to HF hub (qgeng1465/scirecipe-audit).
# Uses the hf-mirror endpoint; token from ~/hf_token.txt (never echoed).
set -e
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="$(tr -d '\r\n ' < /home/qiushuogeng/hf_token.txt)"

.venv/bin/python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
api = HfApi()
repo = "qgeng1465/scirecipe-audit"
try:
    api.repo_info(repo, repo_type="dataset")
except Exception:
    create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    print(f"created {repo}")
for f in ("scirecipe-audit/README.md",
          "results/scirecipe_audit_enriched.json",
          "results/scirecipe_provenance.csv"):
    if not os.path.exists(f):
        print(f"!! missing {f} — skipped")
        continue
    api.upload_file(path_or_fileobj=f, path_in_repo=f.split("/", 1)[1],
                    repo_id=repo, repo_type="dataset")
    print(f"uploaded {f}")
print("done")
PY
