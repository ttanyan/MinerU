<template>
  <div class="mindmap-container">
    <div class="mindmap-header">
      <div class="header-left">
        <span class="mindmap-subtitle" v-if="content">已生成 {{ nodeCount }} 个节点</span>
      </div>
      <div class="mindmap-actions">
        <el-dropdown @command="handleDownload">
          <el-button 
            type="primary" 
            size="small"
            class="action-button primary"
          >
            <template #icon>
              <el-icon><Download /></el-icon>
            </template>
            下载
            <el-icon class="el-icon--right"><ArrowDown /></el-icon>
          </el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="svg">SVG 格式</el-dropdown-item>
              <el-dropdown-item command="png">PNG 格式</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
        <el-button 
          type="default" 
          size="small"
          @click="resetView"
          class="action-button secondary"
        >
          <template #icon>
            <el-icon><Refresh /></el-icon>
          </template>
          重置视图
        </el-button>
      </div>
    </div>
    <div class="mindmap-content" @wheel.prevent="handleWheel">
      <div class="empty-state" v-if="!content">
        <el-icon class="empty-icon"><Document /></el-icon>
        <p class="empty-text">暂无思维导图内容</p>
        <p class="empty-subtext">请先上传并转换文档</p>
      </div>
      <svg ref="svgRef" class="markmap-svg" v-else></svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, onUnmounted, nextTick } from 'vue'
import { Download, Refresh, ZoomIn, ZoomOut, Document, ArrowDown } from '@element-plus/icons-vue'

// 导入 markmap 相关包
import { Transformer } from 'markmap-lib'
import { Markmap, loadCSS, loadJS } from 'markmap-view'

const props = defineProps({
  content: {
    type: String,
    default: ''
  }
})

const svgRef = ref<SVGElement | null>(null)
let mmInstance: any = null
const transformer = new Transformer()
const scale = ref(1)

const nodeCount = computed(() => {
  if (!props.content) return 0
  // 简单计算节点数量（基于Markdown标题）
  return (props.content.match(/^#{1,6}\s+/gm) || []).length
})

const initMarkmap = async () => {
  if (!props.content || !svgRef.value) return

  try {
    console.log('Initializing markmap with content:', props.content.substring(0, 100) + '...')
    
    // 1. 转换数据
    const { root, features } = transformer.transform(props.content)
    console.log('Transformed root:', root)

    // 2. 加载必要的资源
    const { styles, scripts } = transformer.getAssets()
    console.log('Styles:', styles)
    console.log('Scripts:', scripts)
    
    if (styles) loadCSS(styles)
    if (scripts) loadJS(scripts)

    // 3. 创建或更新实例
    await nextTick()
    console.log('SVG ref:', svgRef.value)
    console.log('Markmap.create:', Markmap.create)
    
    if (mmInstance) {
      console.log('Updating existing instance')
      if (typeof mmInstance.setData === 'function') {
        mmInstance.setData(root)
        mmInstance.fit()
      } else {
        console.error('mmInstance does not have setData method:', mmInstance)
        // 创建新实例
        mmInstance = Markmap.create(svgRef.value, {
          autoFit: true,
          fitRatio: 0.9,
          initialExpandLevel: -1
        }, root)
        console.log('Created new instance after setData error:', mmInstance)
      }
    } else {
      console.log('Creating new instance')
      mmInstance = Markmap.create(svgRef.value, {
        autoFit: true,
        fitRatio: 0.9,
        initialExpandLevel: -1
      }, root)
      console.log('Created instance using Markmap.create:', mmInstance)
      console.log('mmInstance methods:', Object.keys(mmInstance))
    }
  } catch (error) {
    console.error('Error initializing markmap:', error)
    mmInstance = null
  }
}

// 监听内容变化自动重绘
watch(() => props.content, () => {
  initMarkmap()
})

onMounted(() => {
  initMarkmap()
  // 窗口缩放时自动调整导图大小
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  if (mmInstance && typeof mmInstance.destroy === 'function') {
    mmInstance.destroy()
  } else if (mmInstance) {
    console.error('mmInstance does not have destroy method:', mmInstance)
  }
})

const handleResize = () => {
  mmInstance?.fit()
}

// 下载思维导图
const downloadMindMap = (format: 'svg' | 'png' = 'svg') => {
  if (!svgRef.value) return
  
  // 确保思维导图已经完全渲染
  if (mmInstance) {
    mmInstance.fit()
  }
  
  if (format === 'svg') {
    const svg = svgRef.value
    // 复制 SVG 元素以确保获取完整结构
    const svgCopy = svg.cloneNode(true) as SVGElement
    
    // 添加必要的命名空间
    if (!svgCopy.getAttribute('xmlns')) {
      svgCopy.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    }
    
    // 确保 SVG 大小正确
    const boundingRect = svg.getBoundingClientRect()
    svgCopy.setAttribute('width', boundingRect.width.toString())
    svgCopy.setAttribute('height', boundingRect.height.toString())
    
    // 序列化 SVG
    const svgData = new XMLSerializer().serializeToString(svgCopy)
    const blob = new Blob([svgData], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `mindmap_${new Date().getTime()}.svg`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  } else if (format === 'png') {
    const svg = svgRef.value
    // 复制 SVG 元素以确保获取完整结构
    const svgCopy = svg.cloneNode(true) as SVGElement
    
    // 添加必要的命名空间
    if (!svgCopy.getAttribute('xmlns')) {
      svgCopy.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    }
    
    // 确保 SVG 大小正确
    const boundingRect = svg.getBoundingClientRect()
    svgCopy.setAttribute('width', boundingRect.width.toString())
    svgCopy.setAttribute('height', boundingRect.height.toString())
    
    // 序列化 SVG
    const svgData = new XMLSerializer().serializeToString(svgCopy)
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    
    if (!ctx) return
    
    // 设置画布大小
    canvas.width = boundingRect.width
    canvas.height = boundingRect.height
    
    // 创建一个图像对象
    const img = new Image()
    img.onload = () => {
      // 绘制图像到画布
      ctx.drawImage(img, 0, 0)
      
      // 将画布转换为 PNG
      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `mindmap_${new Date().getTime()}.png`
          document.body.appendChild(a)
          a.click()
          document.body.removeChild(a)
          URL.revokeObjectURL(url)
        }
      }, 'image/png')
    }
    
    // 将 SVG 数据转换为 Data URL
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgData)
  }
}

// 处理下载命令
const handleDownload = (command: string) => {
  if (command === 'svg' || command === 'png') {
    downloadMindMap(command)
  }
}

// 重置视图
const resetView = () => {
  mmInstance?.fit()
  scale.value = 1
}

// 放大
const zoomIn = () => {
  if (scale.value < 2) {
    scale.value += 0.1
    applyZoom()
  }
}

// 缩小
const zoomOut = () => {
  if (scale.value > 0.5) {
    scale.value -= 0.1
    applyZoom()
  }
}

// 应用缩放
const applyZoom = () => {
  if (mmInstance && typeof mmInstance.setScale === 'function') {
    mmInstance.setScale(scale.value)
  } else if (mmInstance) {
    console.error('mmInstance does not have setScale method:', mmInstance)
  }
}

// 处理鼠标滚轮缩放
const handleWheel = (event: WheelEvent) => {
  event.preventDefault()
  const delta = event.deltaY > 0 ? -0.1 : 0.1
  if ((scale.value > 0.5 || delta > 0) && (scale.value < 2 || delta < 0)) {
    scale.value += delta
    applyZoom()
  }
}
</script>

<style scoped>
.mindmap-container {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: #FFFFFF;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.08);
  overflow: hidden;
  min-height: 0;
}

.mindmap-header {
  padding: 16px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #F2F3F5;
  background-color: transparent;
  box-shadow: none;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.mindmap-title {
  font-size: 16px;
  font-weight: 500;
  color: #1D2129;
  margin: 0;
}

.mindmap-subtitle {
  font-size: 12px;
  color: #86909C;
  background-color: #F2F3F5;
  padding: 2px 8px;
  border-radius: 10px;
}

.mindmap-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}

.action-button {
  border-radius: 6px;
  transition: all 0.3s ease;
  height: 32px;
  padding: 0 12px;
}

.action-button.primary {
  background-color: #165DFF;
  border-color: #165DFF;
  color: #FFFFFF;
}

.action-button.primary:hover {
  background-color: #0E42D2;
  border-color: #0E42D2;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(22, 93, 255, 0.3);
}

.action-button.secondary {
  background-color: #FFFFFF;
  border-color: #C9CDD4;
  color: #4E5969;
}

.action-button.secondary:hover {
  border-color: #165DFF;
  color: #165DFF;
  transform: translateY(-1px);
}

.mindmap-content {
  flex: 1;
  overflow: hidden;
  position: relative;
  background-color: #F9FAFC;
  cursor: grab;
  min-height: 0;
}

.mindmap-content:active {
  cursor: grabbing;
}

.empty-state {
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 16px;
}

.empty-icon {
  font-size: 48px;
  color: #C9CDD4;
}

.empty-text {
  font-size: 16px;
  font-weight: 500;
  color: #4E5969;
  margin: 0;
}

.empty-subtext {
  font-size: 14px;
  color: #86909C;
  margin: 0;
}

.markmap-svg {
  width: 100%;
  height: 100%;
  transition: transform 0.2s ease;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .mindmap-header {
    padding: 12px 16px;
  }
  
  .mindmap-title {
    font-size: 14px;
  }
  
  .mindmap-subtitle {
    font-size: 10px;
  }
  
  .action-button {
    padding: 0 8px;
    font-size: 12px;
  }
  
  .mindmap-actions {
    gap: 4px;
  }
}
</style>