FROM python:3.9-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project files
COPY pyproject.toml .
COPY experiments/ ./experiments/

# Install dependencies with uv
RUN uv pip install --system -e .

# Download nuScenes mini dataset (optional - can be mounted as volume)
# RUN mkdir -p /data/nuscenes

CMD ["/bin/bash"]
