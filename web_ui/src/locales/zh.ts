export default {
  app: {
    title: '多模态思维导图助手'
  },
  common: {
    upload: '上传',
    clear: '清除',
    convert: '转换',
    cancel: '取消',
    confirm: '确定',
    loading: '加载中...',
    success: '成功',
    error: '错误',
    warning: '警告'
  },
  upload: {
    title: '请上传 PDF 或图片',
    placeholder: '点击上传或将文件拖拽到此处',
    supportedTypes: '支持的文件类型: PDF, PNG, JPG, JPEG',
    maxSize: '最大文件大小: 100MB'
  },
  config: {
    title: '配置选项',
    maxPages: '最大转换页数',
    backend: '解析后端',
    serverUrl: '服务器地址',
    serverUrlInfo: 'http-client 后端的 OpenAI 兼容服务器地址。',
    recognitionOptions: '识别选项',
    tableEnable: '启用表格识别',
    tableInfo: '禁用后，表格将显示为图片。',
    formulaLabelVlm: '启用行间公式识别',
    formulaLabelPipeline: '启用公式识别',
    formulaLabelHybrid: '启用行内公式识别',
    formulaInfoVlm: '禁用后，行间公式将显示为图片。',
    formulaInfoPipeline: '禁用后，行间公式将显示为图片，行内公式将不会被检测或解析。',
    formulaInfoHybrid: '禁用后，行内公式将不会被检测或解析。',
    ocrLanguage: 'OCR 语言',
    ocrLanguageInfo: '为扫描版 PDF 和图片选择 OCR 语言。',
    forceOcr: '强制启用 OCR',
    forceOcrInfo: '仅在识别效果极差时启用，需选择正确的 OCR 语言。',
    backendInfoVlm: '多模态大模型高精度解析，仅支持中英文文档。',
    backendInfoPipeline: '传统多模型管道解析，支持多语言，无幻觉。',
    backendInfoHybrid: '高精度混合解析，支持多语言。',
    backendInfoDefault: '选择文档解析的后端引擎。'
  },
  results: {
    title: '转换结果',
    tabs: {
      markdown: 'Markdown 渲染',
      source: 'Markdown 源码',
      mindmap: '思维导图'
    },
    download: '下载结果',
    noResults: '暂无转换结果'
  },
  languages: {
    ch: '中文(简体)',
    en: '英语',
    korean: '韩语',
    japan: '日语',
    chinese_cht: '中文(繁体)',
    ta: '泰米尔语',
    te: '泰卢固语',
    ka: '卡纳达语',
    th: '泰语',
    el: '希腊语',
    latin: '拉丁语系',
    arabic: '阿拉伯语系',
    east_slavic: '东斯拉夫语系',
    cyrillic: '西里尔语系',
    devanagari: '梵文字母语系'
  },
  backends: {
    pipeline: '传统管道解析',
    'vlm-auto-engine': 'VLM本地引擎',
    'hybrid-auto-engine': '混合本地引擎',
    'vlm-http-client': 'VLM远程客户端',
    'hybrid-http-client': '混合远程客户端'
  },
  errors: {
    uploadFailed: '文件上传失败',
    conversionFailed: '转换失败',
    invalidFileType: '不支持的文件类型',
    fileSizeExceeded: '文件大小超出限制',
    networkError: '网络连接错误'
  }
}