export default {
  app: {
    title: 'Multimodal Mind Map Assistant'
  },
  common: {
    upload: 'Upload',
    clear: 'Clear',
    convert: 'Convert',
    cancel: 'Cancel',
    confirm: 'Confirm',
    loading: 'Loading...',
    success: 'Success',
    error: 'Error',
    warning: 'Warning'
  },
  upload: {
    title: 'Please upload PDF or image',
    placeholder: 'Click to upload or drag files here',
    supportedTypes: 'Supported file types: PDF, PNG, JPG, JPEG',
    maxSize: 'Maximum file size: 100MB'
  },
  config: {
    title: 'Configuration Options',
    maxPages: 'Max convert pages',
    backend: 'Backend',
    serverUrl: 'Server URL',
    serverUrlInfo: 'OpenAI-compatible server URL for http-client backend.',
    recognitionOptions: 'Recognition Options',
    tableEnable: 'Enable table recognition',
    tableInfo: 'If disabled, tables will be shown as images.',
    formulaLabelVlm: 'Enable display formula recognition',
    formulaLabelPipeline: 'Enable formula recognition',
    formulaLabelHybrid: 'Enable inline formula recognition',
    formulaInfoVlm: 'If disabled, display formulas will be shown as images.',
    formulaInfoPipeline: 'If disabled, display formulas will be shown as images, and inline formulas will not be detected or parsed.',
    formulaInfoHybrid: 'If disabled, inline formulas will not be detected or parsed.',
    ocrLanguage: 'OCR Language',
    ocrLanguageInfo: 'Select the OCR language for image-based PDFs and images.',
    forceOcr: 'Force enable OCR',
    forceOcrInfo: 'Enable only if the result is extremely poor. Requires correct OCR language.',
    backendInfoVlm: 'High-precision parsing via VLM, supports Chinese and English documents only.',
    backendInfoPipeline: 'Traditional Multi-model pipeline parsing, supports multiple languages, hallucination-free.',
    backendInfoHybrid: 'High-precision hybrid parsing, supports multiple languages.',
    backendInfoDefault: 'Select the backend engine for document parsing.'
  },
  results: {
    title: 'Convert Result',
    tabs: {
      markdown: 'Markdown Rendering',
      source: 'Markdown Source',
      mindmap: 'Mind Map'
    },
    download: 'Download Result',
    noResults: 'No conversion results yet'
  },
  languages: {
    ch: 'Chinese (Simplified)',
    en: 'English',
    korean: 'Korean',
    japan: 'Japanese',
    chinese_cht: 'Chinese (Traditional)',
    ta: 'Tamil',
    te: 'Telugu',
    ka: 'Kannada',
    th: 'Thai',
    el: 'Greek',
    latin: 'Latin Languages',
    arabic: 'Arabic Languages',
    east_slavic: 'East Slavic Languages',
    cyrillic: 'Cyrillic Languages',
    devanagari: 'Devanagari Languages'
  },
  backends: {
    pipeline: 'Traditional Pipeline Parsing',
    'vlm-auto-engine': 'VLM Local Engine',
    'hybrid-auto-engine': 'Hybrid Local Engine',
    'vlm-http-client': 'VLM Remote Client',
    'hybrid-http-client': 'Hybrid Remote Client'
  },
  errors: {
    uploadFailed: 'File upload failed',
    conversionFailed: 'Conversion failed',
    invalidFileType: 'Unsupported file type',
    fileSizeExceeded: 'File size exceeded limit',
    networkError: 'Network connection error'
  }
}