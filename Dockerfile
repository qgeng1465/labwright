# Labwright — one-click runtime image.
#
# Build:    docker build -t labwright .
# Run the benchmark (needs the LLM API key mounted in as env):
#   docker run --rm -e LABWRIGHT_MODEL=deepseek-v4-flash -e LABWRIGHT_BASE_URL=https://... \
#     -e LABWRIGHT_API_KEY=$LABWRIGHT_API_KEY \
#     -v "$PWD/results":/app/results labwright \
#     python -m eval.run_benchmark --systems bare,code_interpreter,labwright --limit 20 --out results/repro.json
#
# The eval runs against the live LLM API; the calculators, the verifier and the
# figure pipeline run entirely inside the image. No wet-lab, no GPU required
# (the fine-tuned extractor stack is optional — see requirements-eval.txt).

FROM python:3.10-slim

# System deps: git for package metadata; ca-certificates for HTTPS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime deps first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package and eval/paper tooling.
COPY labwright ./labwright
COPY eval ./eval
COPY paper ./paper
COPY pyproject.toml .
RUN pip install --no-cache-dir --no-deps .

# Non-root user.
RUN useradd --create-home --uid 1000 labwright
RUN mkdir -p /app/results && chown -R labwright:labwright /app
USER labwright

ENV PYTHONPATH=/app
CMD ["python", "-c", "import labwright; print('Labwright ready. Run eval.run_benchmark / eval.run_adversarial.')"]
