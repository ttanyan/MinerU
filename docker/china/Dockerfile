# ==========================================
# 阶段 1: 构建阶段 (Builder)
# ==========================================
FROM docker.m.daocloud.io/ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 替换为阿里云镜像源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装编译环境、Node.js 及字体依赖
RUN apt-get update && \
    apt-get install -y \
        build-essential curl wget git fontconfig libgl1 \
        libreoffice-writer libreoffice-core \
        fonts-noto-core fonts-noto-cjk \
        python3 python3-pip python3-venv && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# 1. 升级基础 Python 构建工具
RUN pip3 install --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

# 2. 预装指定的最新版核心依赖 (解决版本冲突)
RUN pip3 install \
    "torch>=2.6.0" \
    "torchvision" \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install \
    "modelscope>=1.26.0" \
    "huggingface-hub>=0.32.4" \
    "mineru-vl-utils>=0.1.19.1" \
    "qwen-vl-utils>=0.0.14" \
    "transformers>=4.51.1" \
    "accelerate>=1.5.1" \
    -i https://mirrors.aliyun.com/pypi/simple/

# 3. 安装项目及所有可选依赖 [all]
# 这将自动涵盖 doclayout_yolo==0.0.4 以及图片中显示的 layout/vlm 等所需依赖
RUN pip3 install -e ".[all]" -i https://mirrors.aliyun.com/pypi/simple/

# 4. 构建阶段预下载所有权重文件 (保证离线可用)
# 使用项目自带的脚本下载 layout, ocr, mfd, mfr, vlm 等模型
RUN mkdir -p /root/.cache/modelscope/hub/models && \
    echo '{"models-dir": "/root/.cache/modelscope/hub/models", "device-mode":"cpu"}' > /root/magic-pdf.json && \
    export MINERU_CONFIG_PATH=/root/magic-pdf.json && \
    # 强制输入 'modelscope'
    echo "modelscope" | PYTHONPATH=/app python3 -m mineru.cli.models_download

# 5. 构建前端 Web UI
WORKDIR /app/web_ui
RUN npm install && npm run build
WORKDIR /app
RUN mkdir -p mineru/cli/static/web && cp -r web_ui/dist/* mineru/cli/static/web/

# ==========================================
# 阶段 2: 运行阶段 (Runtime)
# ==========================================
FROM docker.m.daocloud.io/ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y libgl1 libreoffice-writer libreoffice-core \
    fonts-noto-core fonts-noto-cjk fontconfig python3 python3-pip && \
    fc-cache -fv && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从构建阶段拷贝依赖、脚本和预下载的模型
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/modelscope/hub/models /root/.cache/modelscope/hub/models
COPY --from=builder /app /app

# 核心修复：注入标准的运行时配置文件，解决 AttributeError
RUN mkdir -p /root/ && \
    echo '{ \
        "models-dir": "/root/.cache/modelscope/hub/models", \
        "device-mode": "cpu", \
        "vlm-config": { \
            "kind": "transformers", \
            "precision": "fp16" \
        } \
    }' > /root/magic-pdf.json

EXPOSE 8000

# 启动服务，明确设置 PYTHONPATH
ENTRYPOINT ["/bin/sh", "-c", "PYTHONPATH=/app python3 -m mineru.cli.fast_api --host 0.0.0.0 --port 8000"]