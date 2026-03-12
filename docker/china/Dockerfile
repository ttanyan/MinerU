# =====================================================
# Stage 1: Build frontend
# =====================================================
FROM node:20-alpine AS frontend-builder

# Configure npm mirror for China region
RUN npm config set registry https://registry.npmmirror.com

WORKDIR /app/web_ui

# Copy package files
COPY web_ui/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY web_ui/ ./

# Build frontend
RUN npm run build

# =====================================================
# Stage 2: Build backend
# =====================================================
FROM docker.m.daocloud.io/ubuntu:22.04 AS backend-builder

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
        python3-venv && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN python3 -m pip install -U pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple

WORKDIR /app

# Copy project files
COPY . /app

# Install mineru
RUN python3 -m pip install -e '.[core]' -i https://mirrors.aliyun.com/pypi/simple

# Download models
RUN /bin/bash -c "mineru-models-download -s modelscope -m all"

# =====================================================
# Stage 3: Final image
# =====================================================
FROM docker.m.daocloud.io/ubuntu:22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Configure China region mirrors
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 \
        libreoffice-writer \
        libreoffice-core \
        python3 \
        python3-pip && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies
RUN python3 -m pip install -U pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip install -e '.[core]' -i https://mirrors.aliyun.com/pypi/simple

WORKDIR /app

# Copy backend files
COPY --from=backend-builder /app/mineru /app/mineru
COPY --from=backend-builder /app/.mineru_models /app/.mineru_models

# Copy frontend build files
COPY --from=frontend-builder /app/web_ui/dist /app/mineru/cli/static/web

# Expose port
EXPOSE 8000

# Set environment variables
ENV MINERU_MODEL_SOURCE=local

# Entry point
ENTRYPOINT ["python3", "-m", "mineru.cli.fast_api", "--host", "0.0.0.0", "--port", "8000"]
