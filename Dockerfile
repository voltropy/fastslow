FROM volta-runtime:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY artifacts/.gitkeep /app/artifacts/.gitkeep

RUN mkdir -p /app/artifacts \
    && python -m pip install --upgrade pip \
    && python -m pip install -e /app

CMD ["python", "-m", "fastslow.train", "--help"]
