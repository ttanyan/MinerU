# 构建阶段
FROM node:20-alpine as build-stage

WORKDIR /app

# 使用国内镜像源加速
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories

# 复制 package 文件
COPY package*.json ./

# 安装依赖（使用 npm 淘宝镜像）
RUN npm config set registry https://registry.npmmirror.com && \
    npm install

# 复制源代码
COPY . .

# 构建生产版本
RUN npm run build

# 生产阶段
FROM nginx:alpine as production-stage

# 复制 nginx 配置文件
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 复制构建好的静态文件
COPY --from=build-stage /app/dist /usr/share/nginx/html

# 暴露端口
EXPOSE 9898

# 启动 nginx
CMD ["nginx", "-g", "daemon off;"]
