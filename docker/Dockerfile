FROM python:3.11.9-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 tts
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --upgrade pip && python -m pip install .
COPY docker/entrypoint.sh /usr/local/bin/tts-entrypoint
RUN chmod 0555 /usr/local/bin/tts-entrypoint && mkdir -p /app/artifacts && chown -R tts:tts /app/artifacts
USER tts
EXPOSE 8000
ENTRYPOINT ["tts-entrypoint"]
CMD ["tts", "serve", "--config", "configs/development.yaml"]

