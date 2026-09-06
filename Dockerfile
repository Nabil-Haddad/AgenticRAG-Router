FROM python:3.12-slim

WORKDIR /app

# system packages needed by PyMuPDF/onnxruntime at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# torch's CPU-only wheel isn't on plain PyPI, so this build pulls from
# PyTorch's own CPU index in addition to PyPI for everything else
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]
