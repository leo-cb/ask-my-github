# dockerfile to create the demo dashboard image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8505

WORKDIR /app

# libgomp1 is required at runtime by onnxruntime (FastEmbed) if the local
# embedding provider is selected.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY ask_my_github ./ask_my_github
COPY .streamlit ./.streamlit

RUN pip install .

EXPOSE 8505

CMD ["python", "-m", "ask_my_github.dashboard.entrypoint"]
