FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc=4:12.2.0-3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir pytest==8.3.3 pytest-cov==5.0.0

CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=dravex", "--cov-branch"]
