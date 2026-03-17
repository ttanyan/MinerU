# Build stage (构建阶段)
FROM docker.m.daocloud.io/ubuntu:22.04 AS builder

# 设置非交互模式 [cite: 1]
ENV DEBIAN_FRONTEND=noninteractive

# 配置国内镜像源 [cite: 1]
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装构建所需的系统依赖 [cite: 1, 2]
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
    # 安装 Node.js 18 [cite: 2, 3]
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 配置 npm 镜像 [cite: 3]
RUN npm config set registry https://registry.npmmirror.com

WORKDIR /app

# --- 关键修复：先复制依赖描述文件，以便利用缓存 ---
# 确保项目根目录下有这些文件
COPY setup.py* pyproject.toml* README.md* /app/
# 必须先复制源码目录，否则 -e (editable) 安装会因为找不到模块而失败
COPY mineru /app/mineru

# 安装 Python 依赖 [cite: 4]
RUN python3 -m pip install -U pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip install -e '.[core]' -i https://mirrors.aliyun.com/pypi/simple

# 下载模型 [cite: 4]
RUN /bin/bash -c "mineru-models-download -s modelscope -m all"

# 构建前端 [cite: 4]
COPY web_ui /app/web_ui
WORKDIR /app/web_ui
RUN npm install && \
    npm run build

# 创建静态目录并复制前端文件 [cite: 4]
WORKDIR /app
RUN mkdir -p mineru/cli/static/web && \
    cp -r web_ui/dist/* mineru/cli/static/web/

# Runtime stage (运行阶段)
FROM docker.m.daocloud.io/ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# 配置国内镜像源 [cite: 4]
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装运行依赖 [cite: 5]
# 注意：添加了 build-essential 以防某些包在运行时阶段仍需编译
RUN apt-get update && \
    apt-get install -y \
        build-essential \
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
    rm -rf /var/lib/apt/lists/* [cite: 6]

WORKDIR /app

# 从构建阶段拷贝所有内容（包括已安装的库）
COPY --from=builder /app /app
# 拷贝 python 路径下的库文件，确保环境一致
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 在运行阶段重新执行一次轻量安装，以确保可执行命令路径正确
RUN python3 -m pip install -e '.[core]' -i https://mirrors.aliyun.com/pypi/simple

# 暴露端口 [cite: 6]
EXPOSE 8000

# 设置环境变量 [cite: 6]
ENV MINERU_MODEL_SOURCE=local

# 启动命令 [cite: 6]
ENTRYPOINT ["python3", "-m", "mineru.cli.fast_api", "--host", "0.0.0.0", "--port", "8000"]