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

export interface ProgressUpdate {
  progress: number
  status: string
}

export const documentApi = {
  /**
   * 解析文档
   */
  parseDocument(params: ParseParams, onProgress?: (update: ProgressUpdate) => void): Promise<ParseResult> {
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
    
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', 'http://localhost:8000/file_parse')
      
      xhr.onprogress = (event) => {
        if (event.target && event.target.responseText) {
          const responseText = event.target.responseText
          const lines = responseText.split('\n')
          
          for (const line of lines) {
            if (line.trim()) {
              try {
                const data = JSON.parse(line)
                if (data.progress !== undefined && data.status !== undefined) {
                  // 进度更新
                  if (onProgress) {
                    onProgress(data)
                  }
                } else if (data.error) {
                  // 错误信息
                  reject(new Error(data.error))
                } else if (data.results) {
                  // 最终结果
                  resolve(data)
                }
              } catch (e) {
                // 忽略解析错误
              }
            }
          }
        }
      }
      
      xhr.onload = () => {
        if (xhr.status === 200) {
          const responseText = xhr.responseText
          const lines = responseText.split('\n')
          
          for (const line of lines) {
            if (line.trim()) {
              try {
                const data = JSON.parse(line)
                if (data.results) {
                  resolve(data)
                  return
                }
              } catch (e) {
                // 忽略解析错误
              }
            }
          }
          
          reject(new Error('Invalid response format'))
        } else {
          reject(new Error(`Request failed with status ${xhr.status}`))
        }
      }
      
      xhr.onerror = () => {
        reject(new Error('Network error'))
      }
      
      xhr.send(formData)
    })
  }
}