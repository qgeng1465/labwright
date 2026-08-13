# Publishing the Labwright Space

Two deployment states, both under the `qgeng1465` account:

## Current: static showcase Space (free, live)

Hugging Face's 2026 policy no longer lets free accounts create Gradio/Docker
Spaces on `cpu-basic` **or** ZeroGPU — both require a PRO subscription
(`canPay` must be true). Static Spaces stay free, so the live Space is a
static showcase: [huggingface.co/spaces/qgeng1465/labwright](https://huggingface.co/spaces/qgeng1465/labwright).

- Source of the live page: `hf_space/static/` (`index.html` + `assets/` figures +
  `README.md` with `sdk: static` metadata).
- Re-deploy (idempotent) via the Python API — no CLI needed:

```bash
export HF_TOKEN=hf_...  # write token, stored in /home/qiushuogeng/hf_token.txt (never commit)
export HF_ENDPOINT=https://hf-mirror.com  # this host has no direct route to huggingface.co
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi(token=..., endpoint="https://hf-mirror.com")
api.upload_folder(folder_path="hf_space/static",
                  repo_id="qgeng1465/labwright", repo_type="space",
                  commit_message="Update Labwright showcase")
EOF
```

## Future: interactive Gradio Space (requires PRO)

When the account has PRO (`huggingface.co/pro`), switch the Space from static to
Gradio and push the interactive app:

1. Change `hf_space/README.md` metadata: `sdk: gradio`, `sdk_version`, `app_file: app.py`.
2. Push the **root of `hf_space/`** (not `hf_space/static/`) to the Space repo —
   the interactive package lives there (`app.py`, `requirements.txt`, `README.md`).
3. Add a Space secret `DEEPSEEK_API_KEY` (Settings > Secrets) so the design tab
   can call the model. The reverse-verification tab works with no key (pure
   deterministic calculators).

## Files

| path | purpose |
|---|---|
| `hf_space/static/` | live static showcase (sdk: static, free) |
| `hf_space/app.py` | Gradio wrapper around `labwright.ui.app.build_app` (PRO path) |
| `hf_space/requirements.txt` | installs `labwright[agent,ui]` from GitHub (PRO path) |
| `hf_space/README.md` | Gradio Space metadata (title, emoji, SDK) (PRO path) |
