#!/bin/bash

# MinerU Vue.js Web UI 启动脚本

echo "🚀 启动 MinerU Vue.js Web UI..."

# 检查是否已安装依赖
if [ ! -d "node_modules" ]; then
    echo "📦 安装依赖..."
    npm install
fi

# 启动开发服务器
echo "🌐 启动前端开发服务器..."
npm run dev -- --host