import { ref, reactive } from 'vue'
import type { Ref } from 'vue'
import { documentApi, type ParseParams } from '@/api/document'

// 根据上一级标题自动补全下一级标题
function autoPromoteParagraphsToSubheading(text: string): string {
  const lines = text.split('\n')
  const result: string[] = []
  let inSection = false
  let emptyCount = 0
  
  for (const line of lines) {
    const stripped = line.trim()
    
    if (stripped.startsWith('# ')) {
      result.push(line)
      inSection = true
      emptyCount = 0
      continue
    }
    
    if (stripped.startsWith('#')) {
      result.push(line)
      inSection = false
      emptyCount = 0
      continue
    }
    
    if (!stripped) {
      result.push(line)
      emptyCount++
      if (emptyCount >= 2) {
        inSection = false
      }
      continue
    }
    
    // 跳过图片、列表、代码等特殊行
    if (
      stripped.startsWith('![') ||
      stripped.startsWith('>') ||
      stripped.startsWith('```') ||
      /^[-*+] /.test(stripped) ||
      /^\d+\. /.test(stripped)
    ) {
      result.push(line)
      emptyCount = 0
      continue
    }
    
    emptyCount = 0
    if (inSection) {
      result.push('## ' + stripped)
    } else {
      result.push(line)
    }
  }
  
  return result.join('\n')
}

export interface DocumentConfig {
  maxPages: number
  backend: string
  serverUrl: string
  tableEnable: boolean
  formulaEnable: boolean
  language: string
  forceOcr: boolean
}

export interface ProcessResult {
  markdown: string
  source: string
  mindmap: string
  downloadUrl?: string
}

export function useDocumentProcessor() {
  // 文件相关
  const uploadedFiles: Ref<File[]> = ref([])
  const isUploading = ref(false)
  
  // 配置相关
  const config = reactive<DocumentConfig>({
    maxPages: 1000,
    backend: 'hybrid-auto-engine',
    serverUrl: 'http://localhost:30000',
    tableEnable: true,
    formulaEnable: true,
    language: 'ch',
    forceOcr: false
  })
  
  // 结果相关
  const results = ref<ProcessResult | null>(null)
  const isProcessing = ref(false)
  const error = ref<string | null>(null)
  
  // 后端选项
  const backendOptions = [
    { value: 'pipeline', label: '传统管道解析' },
    { value: 'vlm-auto-engine', label: 'VLM本地引擎' },
    { value: 'hybrid-auto-engine', label: '混合本地引擎' },
    { value: 'vlm-http-client', label: 'VLM远程客户端' },
    { value: 'hybrid-http-client', label: '混合远程客户端' }
  ]
  
  // 语言选项
  const languageOptions = [
    { value: 'ch', label: '中文(简体)' },
    { value: 'en', label: '英语' },
    { value: 'korean', label: '韩语' },
    { value: 'japan', label: '日语' },
    { value: 'chinese_cht', label: '中文(繁体)' },
    { value: 'ta', label: '泰米尔语' },
    { value: 'te', label: '泰卢固语' },
    { value: 'ka', label: '卡纳达语' },
    { value: 'th', label: '泰语' },
    { value: 'el', label: '希腊语' },
    { value: 'latin', label: '拉丁语系' },
    { value: 'arabic', label: '阿拉伯语系' },
    { value: 'east_slavic', label: '东斯拉夫语系' },
    { value: 'cyrillic', label: '西里尔语系' },
    { value: 'devanagari', label: '梵文字母语系' }
  ]
  
  // 文件上传处理
  const handleFileUpload = (files: FileList | null) => {
    if (!files || files.length === 0) return
    
    const validFiles: File[] = []
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      const fileType = file.type
      
      // 验证文件类型
      if (!fileType.startsWith('image/') && fileType !== 'application/pdf') {
        error.value = '不支持的文件类型'
        continue
      }
      
      // 验证文件大小 (100MB)
      if (file.size > 100 * 1024 * 1024) {
        error.value = '文件大小超出限制'
        continue
      }
      
      validFiles.push(file)
    }
    
    uploadedFiles.value = validFiles
    error.value = null
  }
  
  // 清除所有数据
  const clearAll = () => {
    uploadedFiles.value = []
    results.value = null
    error.value = null
  }
  
  // 处理文档转换
  const processDocument = async () => {
    if (uploadedFiles.value.length === 0) {
      error.value = '请先上传文件'
      return
    }
    
    isProcessing.value = true
    error.value = null
    
    try {
      const params: ParseParams = {
        files: uploadedFiles.value,
        output_dir: './output',
        lang_list: [config.language],
        backend: config.backend,
        parse_method: config.forceOcr ? 'ocr' : 'auto',
        formula_enable: config.formulaEnable,
        table_enable: config.tableEnable,
        start_page_id: 0,
        end_page_id: config.maxPages - 1,
        return_md: true,
        return_middle_json: false,
        response_format_zip: false
      }
      
      if (config.backend.includes('http-client') && config.serverUrl) {
        params.server_url = config.serverUrl
      }
      
      const response = await documentApi.parseDocument(params)
      
      if (response.results) {
        const resultData = Object.values(response.results)[0]
        const mdContent = resultData.md_content || ''
        const processedSource = autoPromoteParagraphsToSubheading(mdContent)
        results.value = {
          markdown: mdContent,
          source: processedSource,
          mindmap: mdContent
        }
      }
      
    } catch (err: any) {
      error.value = err.message || '转换失败'
    } finally {
      isProcessing.value = false
    }
  }
  
  // 根据后端类型获取公式标签
  const getFormulaLabel = (backend: string) => {
    if (backend.startsWith('vlm')) {
      return '启用行间公式识别'
    } else if (backend === 'pipeline') {
      return '启用公式识别'
    } else if (backend.startsWith('hybrid')) {
      return '启用行内公式识别'
    }
    return '启用公式识别'
  }
  
  // 根据后端类型获取公式说明
  const getFormulaInfo = (backend: string) => {
    if (backend.startsWith('vlm')) {
      return '禁用后，行间公式将显示为图片。'
    } else if (backend === 'pipeline') {
      return '禁用后，行间公式将显示为图片，行内公式将不会被检测或解析。'
    } else if (backend.startsWith('hybrid')) {
      return '禁用后，行内公式将不会被检测或解析。'
    }
    return ''
  }
  
  return {
    // 数据
    uploadedFiles,
    config,
    results,
    isUploading,
    isProcessing,
    error,
    
    // 选项
    backendOptions,
    languageOptions,
    
    // 方法
    handleFileUpload,
    clearAll,
    processDocument,
    getFormulaLabel,
    getFormulaInfo
  }
}