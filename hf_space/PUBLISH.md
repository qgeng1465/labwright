# Publishing the Labwright Space

`hf_space/` is ready to deploy as a Hugging Face Space (Gradio SDK). You need a
Hugging Face account with a write token (`hf_...`). This repo does not ship a
token; the deploy step below is a 60-second manual action.

## One-shot CLI deploy

```bash
# from this repo
HF_TOKEN=hf_...  # your write token (never commit this)
cd hf_space
pip install -q huggingface_hub

# create the Space (skip if it exists)
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi(token="hf_...")
api.create_repo(repo_id="qgeng1465/labwright", repo_type="space", private=False)
EOF

# push files
huggingface-cli upload qgeng1465/labwright . --repo-type=space \
  --commit-message="Labwright Space (v0.1)"

# after first build, add a Space secret (Settings > Secrets):
#   DEEPSEEK_API_KEY = your OpenAI-compatible key  -> enables the design tab
```

The reverse-verification tab works with no key (pure calculators).

## Files

| file | purpose |
|---|---|
| `app.py` | thin wrapper around `labwright.ui.app.build_app` |
| `requirements.txt` | installs `labwright[agent,ui]` from GitHub |
| `README.md` | Space metadata (title, emoji, SDK) shown on the hub |
