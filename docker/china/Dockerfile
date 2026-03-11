# Use Ubuntu 22.04 as base image
FROM docker.m.daocloud.io/ubuntu:22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Configure China region mirrors
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        curl \
        wget \
        git \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 \
        libreoffice-writer \
        libreoffice-core \
        python3 \
        python3-pip \
        python3-venv \
        nodejs \
        npm && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure npm mirror for China region
RUN npm config set registry https://registry.npmmirror.com

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN python3 -m pip install -U pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip install -e '.[core]' -i https://mirrors.aliyun.com/pypi/simple

# Download models
RUN /bin/bash -c "mineru-models-download -s modelscope -m all"

# Build frontend
WORKDIR /app/web_ui
RUN npm install && \
    npm run build

# Create static directory for frontend
WORKDIR /app
RUN mkdir -p mineru/cli/static/web && \
    cp -r web_ui/dist/* mineru/cli/static/web/

# Expose ports
EXPOSE 8000

# Set environment variables
ENV MINERU_MODEL_SOURCE=local

# Set entry point to start FastAPI server which serves both API and static files
ENTRYPOINT ["python3", "-m", "mineru.cli.fast_api", "--host", "0.0.0.0", "--port", "8000"]
