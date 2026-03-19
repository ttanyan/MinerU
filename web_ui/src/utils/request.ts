import axios from 'axios'

// 创建 axios 实例
const request = axios.create({
  baseURL: '', // 空字符串，使用相对路径
  timeout: 300000, // 5分钟超时
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
request.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证token等
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
request.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    // 统一错误处理
    if (error.response) {
      switch (error.response.status) {
        case 400:
          error.message = '请求参数错误'
          break
        case 401:
          error.message = '未授权，请重新登录'
          break
        case 403:
          error.message = '拒绝访问'
          break
        case 404:
          error.message = '请求资源不存在'
          break
        case 500:
          error.message = '服务器内部错误'
          break
        case 502:
          error.message = '网关错误'
          break
        case 503:
          error.message = '服务不可用'
          break
        case 504:
          error.message = '网关超时'
          break
        default:
          error.message = `连接错误${error.response.status}`
      }
    } else {
      error.message = '网络连接异常'
    }
    
    return Promise.reject(error)
  }
)

export default request