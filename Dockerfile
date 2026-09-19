# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12

# ---- build: wheels for the app and its dependencies -------------------------
FROM python:${PYTHON_VERSION}-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
# CPU only torch keeps the image small; the local model runs on CPU.
RUN pip wheel --wheel-dir /wheels --extra-index-url https://download.pytorch.org/whl/cpu ".[azure,local]"

# ---- runtime ----------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime
ARG LOCAL_MODEL_ID=Qwen/Qwen2.5-Coder-0.5B-Instruct
ARG BAKE_LOCAL_MODEL=true
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SA_ENV=production \
    SA_CONFIG_DIR=/app/configs \
    HF_HOME=/app/.hf \
    HF_HUB_OFFLINE=1

# Microsoft ODBC Driver 18 for Azure SQL
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gnupg ca-certificates apt-transport-https \
 && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
 && apt-get update \
 && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 unixodbc \
 && apt-get purge -y curl gnupg \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY configs ./configs
COPY prompts ./prompts
COPY migrations ./migrations
COPY evaluation ./evaluation

# Bake the local SQL model into the image so pods start without network access.
RUN if [ "$BAKE_LOCAL_MODEL" = "true" ]; then \
      HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; snapshot_download('${LOCAL_MODEL_ID}')"; \
    fi \
 && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "sustainability_advisor.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
