# =====================================================
# Stage 1: Build frontend
# =====================================================
FROM node:20-alpine AS frontend-builder
RUN npm config set registry https://registry.npmmirror.com
WORKDIR /app/web_ui
COPY web_ui/package*.json ./
RUN npm install
COPY web_ui/ ./
RUN npm run build

# =====================================================
# Stage 2: Build backend (Download Models)
# =====================================================
FROM docker.m.daocloud.io/ubuntu:22.04 AS backend-builder
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y \
    build-essential curl wget git python3 python3-pip python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install -U pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple
WORKDIR /app
COPY . /app
# 使用可编辑模式安装，确保路径正确
RUN python3 -m pip install -e '.[core]' -i https://mirrors.aliyun.com/pypi/simple
RUN mkdir -p /app/.mineru_models && \
    /bin/bash -c "mineru-models-download -s modelscope -m all"

# =====================================================
# Stage 3: Final image
# =====================================================
FROM docker.m.daocloud.io/ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# 安装运行时依赖
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y \
    fonts-noto-core fonts-noto-cjk fontconfig libgl1 \
    libreoffice-writer libreoffice-core python3 python3-pip && \
    fc-cache -fv && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple

WORKDIR /app

# 1. 拷贝项目源码
COPY pyproject.toml ./
COPY mineru ./mineru

# 2. 【核心修复】以可编辑模式安装，使 Python 模块指向 /app/mineru
RUN python3 -m pip install -e '.[core]' -i https://mirrors.aliyun.com/pypi/simple

# 3. 拷贝模型
RUN mkdir -p /app/.mineru_models
COPY --from=backend-builder /app/.mineru_models /app/.mineru_models

# 4. 【核心修复】将前端文件拷贝到 /app 内，确保 fast_api.py 能定位
COPY --from=frontend-builder /app/web_ui/dist /app/mineru/cli/static/web

# 设置环境
ENV MINERU_MODEL_SOURCE=local
EXPOSE 8989

# 5. 使用模块化方式启动
ENTRYPOINT ["python3", "-m", "mineru.cli.fast_api", "--host", "0.0.0.0", "--port", "8989"]