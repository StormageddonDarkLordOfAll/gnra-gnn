FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# (Optional) Create a non-root user to match VS Code behavior
RUN useradd -ms /bin/bash vscode
USER vscode
