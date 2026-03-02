# MinerU Vue.js Web UI

基于 Vue 3 + TypeScript + Element Plus 的现代化 Web 界面，用于替代原有的 Gradio 界面。

## 🌟 特性

- ✨ 响应式设计，支持移动端
- 🌍 国际化支持（中英文）
- 📁 文件拖拽上传
- ⚙️ 丰富的配置选项
- 📊 Markdown 渲染和源码查看
- 🧠 思维导图可视化（开发中）
- 🔗 与 FastAPI 后端无缝集成

## 🚀 快速开始

### 环境要求

- Node.js >= 16.0.0
- npm >= 8.0.0

### 安装依赖

```bash
cd web_ui
npm install
```

### 启动开发服务器

```bash
# 方式1：使用 npm
npm run dev

# 方式2：使用启动脚本
./start.sh
```

### 构建生产版本

```bash
npm run build
```

## 🏗️ 项目结构

```
web_ui/
├── src/
│   ├── components/          # 可复用组件
│   │   ├── FileUploader.vue    # 文件上传组件
│   │   ├── ConfigPanel.vue     # 配置面板组件
│   │   ├── ResultPanel.vue     # 结果展示组件
│   │   ├── MarkdownRenderer.vue # Markdown 渲染器
│   │   └── MindMapRenderer.vue  # 思维导图渲染器
│   ├── views/               # 页面视图
│   │   └── DocumentProcessor.vue # 主处理器页面
│   ├── composables/         # Vue 组合式函数
│   │   └── useDocumentProcessor.ts # 文档处理逻辑
│   ├── locales/             # 国际化文件
│   │   ├── zh.ts               # 中文翻译
│   │   ├── en.ts               # 英文翻译
│   │   └── index.ts            # 导出文件
│   ├── api/                 # API 接口
│   │   └── document.ts         # 文档处理 API
│   ├── utils/               # 工具函数
│   │   └── request.ts          # HTTP 请求封装
│   ├── App.vue              # 根组件
│   └── main.ts              # 入口文件
├── public/                  # 静态资源
├── index.html               # HTML 模板
├── vite.config.ts           # Vite 配置
├── tsconfig.json            # TypeScript 配置
├── package.json             # 项目依赖
└── README.md                # 本文档
```

## 🔧 技术栈

- **Vue 3** - 渐进式 JavaScript 框架
- **TypeScript** - JavaScript 的超集，提供类型安全
- **Element Plus** - Vue 3 组件库
- **Vite** - 新一代构建工具
- **Axios** - HTTP 客户端
- **Vue I18n** - 国际化解决方案
- **Markdown-it** - Markdown 解析器

## 🔄 与原 Gradio 界面的区别

| 特性 | 原 Gradio | Vue.js 版本 |
|------|-----------|-------------|
| 用户体验 | 基础交互 | 现代化 UI 设计 |
| 响应式 | 有限支持 | 完全响应式 |
| 国际化 | 有限支持 | 完善的多语言支持 |
| 组件化 | 不支持 | 高度组件化 |
| 扩展性 | 较差 | 易于扩展和维护 |
| 移动端适配 | 不支持 | 响应式设计 |

## 📋 功能对比

### ✅ 已实现功能
- [x] 文件上传（PDF/图片）
- [x] 参数配置面板
- [x] 后端选择（pipeline/VLM/hybrid）
- [x] OCR 语言选择
- [x] 表格和公式识别选项
- [x] Markdown 渲染展示
- [x] Markdown 源码查看
- [x] 结果下载
- [x] 中英文国际化
- [x] 错误处理和提示

### ⏳ 待完善功能
- [ ] 交互式思维导图
- [ ] 批量处理支持
- [ ] 处理进度显示
- [ ] 历史记录管理
- [ ] 用户偏好设置保存
- [ ] 更多主题样式

## 🛠️ 开发指南

### 添加新组件

1. 在 `src/components/` 目录下创建新的 `.vue` 文件
2. 使用 Composition API 编写组件逻辑
3. 在需要的地方导入和使用组件

### 添加国际化文本

1. 在 `src/locales/zh.ts` 和 `src/locales/en.ts` 中添加对应的翻译
2. 在组件中使用 `$t('key')` 来引用翻译文本

### API 接口扩展

1. 在 `src/api/` 目录下创建新的 API 文件
2. 使用 `request.ts` 封装的 axios 实例
3. 在 `composables` 中调用 API

## 🐛 常见问题

### 1. 启动时报错 "Port 3000 is in use"
这是正常的，Vite 会自动寻找可用端口。通常会在 3001、3002 等端口启动。

### 2. 无法连接到后端 API
确保 FastAPI 服务已在 8000 端口启动：
```bash
mineru-api --host localhost --port 8000
```

### 3. 构建失败
尝试清理缓存并重新安装依赖：
```bash
rm -rf node_modules package-lock.json
npm install
npm run build
```

## 📄 许可证

本项目遵循 AGPL-3.0 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**：此 Vue.js 版本目前仍在开发中，部分功能可能不如原 Gradio 版本稳定。建议在生产环境中继续使用原版 Gradio 界面。