FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY factory ./factory
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[dev]"

ENV PYTHONUNBUFFERED=1

CMD ["factory"]
