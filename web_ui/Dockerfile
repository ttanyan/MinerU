# 构建阶段
FROM node:18-alpine AS builder

# 设置工作目录
WORKDIR /app

# 使用国内镜像源加速
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories

# 复制 package.json 和 package-lock.json
COPY package*.json ./

# 安装依赖（使用 npm 淘宝镜像）
RUN npm config set registry https://registry.npmmirror.com && \
    npm install

# 复制源代码
COPY . .

# 构建前端
RUN npm run build

# 运行阶段
FROM nginx:alpine

# 复制构建结果到 Nginx 静态目录
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制 Nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 暴露端口
EXPOSE 9898

# 启动 Nginx
CMD ["nginx", "-g", "daemon off;"]