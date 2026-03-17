# ==========================================
# 阶段 1: 构建阶段 (Builder) - 负责前端 UI 编译
# ==========================================
FROM docker.m.daocloud.io/ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 替换为阿里云镜像源并安装 Node.js [cite: 8, 9]
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

WORKDIR /app
COPY . .

# 构建前端 Web UI [cite: 12]
WORKDIR /app/web_ui
RUN npm install && npm run build


# ==========================================
# 阶段 2: 运行阶段 (Runtime) - 天数智芯专用环境
# ==========================================
# 使用天数智芯官方适配镜像，该镜像内置了 Python 3.10.18 和 CoreX 驱动 [cite: 15]
FROM crpi-vofi3w62lkohhxsp.cn-shanghai.personal.cr.aliyuncs.com/opendatalab-mineru/corex:4.4.0_torch2.7.1_vllm0.11.2_py3.10

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 1. 安装字体与 LibreOffice 依赖 [cite: 15]
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 \
        libreoffice-writer \
        libreoffice-core && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. 核心环境修复：确保在 CoreX 的 Python 路径下安装依赖
# 注意：使用 python3 -m pip 确保安装到 3.10.18 环境，避免 uvicorn 找不到 [cite: 16]
RUN python3 -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip install \
        'mineru[core]>=2.7.4' \
        "uvicorn" \
        "fastapi" \
        "python-multipart" \
        "modelscope>=1.26.0" \
        "huggingface-hub>=0.32.4" \
        "mineru-vl-utils>=0.1.19.1" \
        "qwen-vl-utils>=0.0.14" \
        numpy==1.26.4 \
        opencv-python==4.11.0.86 \
        -i https://mirrors.aliyun.com/pypi/simple

# 3. 拷贝源码及第一阶段的前端产物 [cite: 10, 13]
COPY . .
RUN mkdir -p /app/mineru/cli/static/web && \
    cp -r /app/web_ui/dist/* /app/mineru/cli/static/web/

# 4. 下载模型权重 (离线模式必备) [cite: 16]
RUN /bin/bash -c "mineru-models-download -s modelscope -m all"

# 5. 注入适配天数 GPU 的配置文件 (开启 vLLM 推理) [cite: 13]
RUN mkdir -p /root/ && \
    echo '{ \
        "models-dir": "/root/.cache/modelscope/hub/models", \
        "device-mode": "cuda", \
        "vlm-config": { \
            "kind": "vllm", \
            "precision": "fp16" \
        } \
    }' > /root/magic-pdf.json

# 6. 设置环境变量 [cite: 17]
ENV MINERU_MODEL_SOURCE=local
ENV PYTHONPATH=/app
EXPOSE 8000

# 7. 启动服务：使用 /bin/bash 包装以加载 CoreX 环境路径，解决二进制执行错误
ENTRYPOINT ["/bin/bash", "-c", "PYTHONPATH=/app exec python3 -m mineru.cli.fast_api --host 0.0.0.0 --port 8000"]