# ManaMind Docker Image
# Multi-stage build for efficient training and deployment

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=11.8

# Base image with CUDA support for training
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Create app user
RUN groupadd -r manamind && useradd -r -g manamind -d /app -s /bin/bash manamind

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -e .[training,docker]

# Training stage - includes full dependencies
FROM base as training

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install ManaMind in development mode
RUN pip install -e .

# Download and setup Forge
RUN mkdir -p /app/forge && \
    wget -O /tmp/forge.tar.bz2 "https://releases.cardforge.org/forge/forge-gui-latest.tar.bz2" && \
    tar -xjf /tmp/forge.tar.bz2 -C /app/forge --strip-components=1 && \
    rm /tmp/forge.tar.bz2

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Create data directories
RUN mkdir -p /app/data/{checkpoints,logs,game_logs,cards}

# Change ownership to app user
RUN chown -R manamind:manamind /app

# Switch to app user
USER manamind

# Set default command
CMD ["manamind", "train", "--config", "configs/base.yaml"]

# Evaluation stage - lighter image for inference only
FROM base as evaluation

# Copy only necessary files
COPY src/ ./src/
COPY configs/base.yaml ./configs/
COPY --from=training /app/forge /app/forge

# Install ManaMind
RUN pip install -e .

# Create minimal data directories
RUN mkdir -p /app/data/checkpoints

# Change ownership
RUN chown -R manamind:manamind /app

# Switch to app user
USER manamind

# Set default command
CMD ["manamind", "eval", "--help"]

# Development stage - includes dev tools
FROM training as development

# Switch back to root for dev tool installation
USER root

# Install development dependencies
RUN pip install -e .[dev]

# Install additional dev tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Setup git (for development)
RUN git config --system --add safe.directory /app

# Install Jupyter for interactive development
RUN pip install jupyter jupyterlab notebook

# Switch back to app user
USER manamind

# Expose Jupyter port
EXPOSE 8888

# Set default command
CMD ["bash"]