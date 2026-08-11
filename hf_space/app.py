"""Labwright Hugging Face Space — the web demo.

A thin wrapper: all the logic lives in ``labwright.ui.app`` — the exact same
pipeline the CLI, the notebook and the benchmark use. Add your
``DEEPSEEK_API_KEY`` as a Space secret to enable the agent; the
reverse-verification tab works with no key at all (pure calculators).
"""

from labwright.ui.app import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
