import request from '@/utils/request'
import type { AxiosResponse } from 'axios'

export interface ParseParams {
  files: File[]
  output_dir: string
  lang_list: string
  backend: string
  parse_method: string
  formula_enable: boolean
  table_enable: boolean
  server_url?: string
  start_page_id: number
  end_page_id: number
  return_md: boolean
  return_middle_json: boolean
  response_format_zip: boolean
}

export interface ParseResult {
  backend: string
  version: string
  results: Record<string, {
    md_content?: string
    middle_json?: string
    model_output?: string
    content_list?: string
    images?: Record<string, string>
  }>
}

export const documentApi = {
  /**
   * 解析文档
   */
  parseDocument(params: ParseParams): Promise<ParseResult> {
    const formData = new FormData()
    
    // 添加文件
    params.files.forEach(file => {
      formData.append('files', file)
    })
    
    // 添加其他参数
    formData.append('output_dir', params.output_dir)
    formData.append('lang_list', params.lang_list)
    formData.append('backend', params.backend)
    formData.append('parse_method', params.parse_method)
    formData.append('formula_enable', params.formula_enable.toString())
    formData.append('table_enable', params.table_enable.toString())
    formData.append('start_page_id', params.start_page_id.toString())
    formData.append('end_page_id', params.end_page_id.toString())
    formData.append('return_md', params.return_md.toString())
    formData.append('return_middle_json', params.return_middle_json.toString())
    formData.append('response_format_zip', params.response_format_zip.toString())
    
    if (params.server_url) {
      formData.append('server_url', params.server_url)
    }
    
    return request.post('/file_parse', formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
    }).then(result => {
        console.log("解析成功:", result);
        return result;
    }).catch(error => {
        console.error("解析失败:", error);
        throw error;
    })
  }
}