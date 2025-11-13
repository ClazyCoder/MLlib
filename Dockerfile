FROM pytorch/pytorch:12.1.1-cuda11.8-cudnn8-runtime
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml /app/
COPY src/ /app/src/

# Install dependencies using uv
RUN uv pip install --system -e .

# Copy remaining files
COPY entrypoint.sh /app/

ENTRYPOINT ["bash", "entrypoint.sh"]