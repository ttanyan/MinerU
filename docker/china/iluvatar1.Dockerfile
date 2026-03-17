# ==========================================
# 阶段 1: 构建阶段 (Builder) - 适配天数智芯 (Iluvatar CoreX)
# ==========================================
FROM crpi-vofi3w62lkohhxsp.cn-shanghai.personal.cr.aliyuncs.com/opendatalab-mineru/corex:4.4.0_torch2.7.1_vllm0.11.2_py3.10 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 替换为阿里云镜像源（corex base 已预优化，若无匹配则不影响）
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装构建环境、Node.js（Web UI）、libreoffice、字体等依赖
# corex base 已包含部分字体与 Python，但仍需补充构建工具与 Node.js
RUN apt-get update && \
    apt-get install -y \
        build-essential curl wget git fontconfig libgl1 \
        libreoffice-writer libreoffice-core \
        fonts-noto-core fonts-noto-cjk \
        python3-pip && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# 1. 升级基础 Python 构建工具
RUN python3 -m pip install --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

RUN python3 -m pip install --no-cache-dir \
    "uvicorn[standard]>=0.30" \
    "fastapi>=0.115" \
    "python-multipart>=0.0.9" \
    -i https://mirrors.aliyun.com/pypi/simple/

# 2. 引入 corex.Dockerfile 的 pinned 依赖（解决版本冲突）
RUN python3 -m pip install \
    numpy==1.26.4 \
    opencv-python==4.11.0.86 \
    -i https://mirrors.aliyun.com/pypi/simple/

# 3. 预装项目所需核心依赖（跳过 torch，因为 corex base 已提供 GPU 版）
RUN python3 -m pip install \
    "modelscope>=1.26.0" \
    "huggingface-hub>=0.32.4" \
    "mineru-vl-utils>=0.1.19.1" \
    "qwen-vl-utils>=0.0.14" \
    "transformers>=4.51.1" \
    "accelerate>=1.5.1" \
    -i https://mirrors.aliyun.com/pypi/simple/

# 4. 安装项目及所有可选依赖 [all]（自动涵盖 doclayout_yolo、layout/vlm 等）
RUN python3 -m pip install -e ".[all]" -i https://mirrors.aliyun.com/pypi/simple/

# 5. 构建阶段预下载所有权重文件（结合 corex 的下载命令 + 配置）
RUN mkdir -p /root/.cache/modelscope/hub/models && \
    echo '{"models-dir": "/root/.cache/modelscope/hub/models", "device-mode":"gpu"}' > /root/magic-pdf.json && \
    export MINERU_CONFIG_PATH=/root/magic-pdf.json && \
    /bin/bash -c "mineru-models-download -s modelscope -m all"

# 6. 构建前端 Web UI
WORKDIR /app/web_ui
RUN npm install && npm run build
WORKDIR /app
RUN mkdir -p mineru/cli/static/web && cp -r web_ui/dist/* mineru/cli/static/web/

# ==========================================
# 阶段 2: 运行阶段 (Runtime)
# ==========================================
FROM crpi-vofi3w62lkohhxsp.cn-shanghai.personal.cr.aliyuncs.com/opendatalab-mineru/corex:4.4.0_torch2.7.1_vllm0.11.2_py3.10 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# 替换为阿里云镜像源（安全起见）
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list || true

RUN apt-get update && \
    apt-get install -y libgl1 libreoffice-writer libreoffice-core \
    fonts-noto-core fonts-noto-cjk fontconfig python3 python3-pip && \
    fc-cache -fv && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从构建阶段拷贝依赖、预下载模型、源码和 Web UI 静态文件
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/modelscope/hub/models /root/.cache/modelscope/hub/models
COPY --from=builder /app /app

# 核心修复：注入标准的运行时配置文件（适配 GPU）
RUN mkdir -p /root/ && \
    echo '{ \
        "models-dir": "/root/.cache/modelscope/hub/models", \
        "device-mode": "gpu", \
        "vlm-config": { \
            "kind": "transformers", \
            "precision": "fp16" \
        } \
    }' > /root/magic-pdf.json

EXPOSE 8000

# 启动服务（结合 corex 的 MINERU_MODEL_SOURCE=local + 原 fast_api 入口）
ENTRYPOINT ["/bin/sh", "-c", "export MINERU_MODEL_SOURCE=local && PYTHONPATH=/app python3 -m mineru.cli.fast_api --host 0.0.0.0 --port 8000"]