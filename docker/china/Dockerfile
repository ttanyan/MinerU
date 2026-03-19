# ==========================================
# 阶段 1：前端构建 (Node.js Builder)
# ==========================================
FROM node:18-alpine AS frontend-builder

WORKDIR /build
# 利用镜像缓存：先拷贝 package.json
COPY web_ui/package*.json ./
RUN npm config set registry https://registry.npmmirror.com && \
    npm install

# 拷贝源码并构建
COPY web_ui/ .
RUN npm run build

# ==========================================
# 阶段 2：最终运行环境 (CoreX Runtime)
# ==========================================
FROM crpi-vofi3w62lkohhxsp.cn-shanghai.personal.cr.aliyuncs.com/opendatalab-mineru/corex:4.4.0_torch2.7.1_vllm0.11.2_py3.10

ENV DEBIAN_FRONTEND=noninteractive
ENV MINERU_MODEL_SOURCE=local

# 1. 系统级依赖：仅安装运行时必需的字体和工具
RUN apt-get update && apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1-mesa-glx \
    && fc-cache -fv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python 依赖安装策略：
# 使用 --no-deps 确保不覆盖 CoreX 预装的 torch 2.7.1 和 vllm 0.11.2
RUN python3 -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip install 'mineru[core]>=2.7.4' --no-deps -i https://mirrors.aliyun.com/pypi/simple && \
    # 手动补齐 MinerU 运行必需但基础镜像可能缺失的轻量依赖（不会破坏 torch）
    python3 -m pip install \
        "numpy==1.26.4" \
        "opencv-python==4.11.0.86" \
        "pydantic<2.0" \
        "modelscope" \
        "magic-pdf" \
        -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip cache purge

WORKDIR /app

# 3. 拷贝项目源码（排除 web_ui 源码以减小体积）
COPY . .

# ... 前面步骤保持不变 ...

# 4. 从阶段 1 拷贝前端构建产物
RUN mkdir -p mineru/cli/static/web
COPY --from=frontend-builder /build/dist/ ./mineru/cli/static/web/

# 5. 配置文件优化：必须先于模型下载执行
RUN mkdir -p /root/.cache/modelscope/hub/models && \
    echo '{ \
        "models-dir": "/root/.cache/modelscope/hub/models", \
        "device-mode": "gpu", \
        "vlm-config": { \
            "kind": "vllm", \
            "precision": "fp16" \
        } \
    }' > /root/magic-pdf.json

# 6. 模型预下载
RUN /bin/bash -c "export MINERU_MODEL_SOURCE=local && mineru-models-download -s modelscope -m all"



# 7. 入口点
ENTRYPOINT ["/bin/bash", "-c", "exec \"$@\"", "--"]
CMD ["python3", "-m", "mineru.cli.main"]