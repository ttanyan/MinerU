<template>
  <div class="mindmap-container">
    <div class="mindmap-header">
      <div class="header-left">
        <h3 class="mindmap-title">思维导图</h3>
        <span class="mindmap-subtitle" v-if="content">已生成 {{ nodeCount }} 个节点</span>
      </div>
      <div class="mindmap-actions">
        <el-button 
          type="primary" 
          size="small"
          @click="downloadMindMap"
          class="action-button primary"
        >
          <template #icon>
            <el-icon><Download /></el-icon>
          </template>
          下载
        </el-button>
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
        <el-button 
          type="default" 
          size="small"
          @click="zoomIn"
          class="action-button secondary"
        >
          <template #icon>
            <el-icon><ZoomIn /></el-icon>
          </template>
          放大
        </el-button>
        <el-button 
          type="default" 
          size="small"
          @click="zoomOut"
          class="action-button secondary"
        >
          <template #icon>
            <el-icon><ZoomOut /></el-icon>
          </template>
          缩小
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
import { Transformer } from 'markmap-lib'
import { loadCSS, loadJS } from 'markmap-view'
import { Download, Refresh, ZoomIn, ZoomOut, Document } from '@element-plus/icons-vue'

// 为 window.Markmap 添加类型声明
declare global {
  interface Window {
    Markmap: any
  }
}

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
    console.log('Loading assets:', { styles: !!styles, scripts: !!scripts })
    if (styles) loadCSS(styles)
    if (scripts) {
      await loadJS(scripts)
      // 3. 创建或更新实例
      await nextTick()
      console.log('SVG ref:', svgRef.value)
      
      // 使用全局 Markmap 对象
      if (window.Markmap) {
        if (mmInstance) {
          console.log('Updating existing instance')
          mmInstance.setData(root)
          mmInstance.fit()
        } else {
          console.log('Creating new instance')
          mmInstance = window.Markmap.create(svgRef.value, {
            autoFit: true,
            fitRatio: 0.9,
            initialExpandLevel: -1,
            color: {
              primary: '#165DFF',
              secondary: '#4E5969',
              tertiary: '#86909C'
            },
            padding: 60,
            nodePadding: 12,
            lineWidth: 2,
            spacingVertical: 40,
            spacingHorizontal: 60
          }, root)
          console.log('Created instance:', mmInstance)
        }
      } else {
        console.error('Markmap is not loaded')
      }
    }
  } catch (error) {
    console.error('Error initializing markmap:', error)
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
  if (mmInstance) {
    mmInstance.destroy()
  }
})

const handleResize = () => {
  mmInstance?.fit()
}

// 下载思维导图
const downloadMindMap = () => {
  if (!svgRef.value) return
  
  const svg = svgRef.value
  const svgData = new XMLSerializer().serializeToString(svg)
  const blob = new Blob([svgData], { type: 'image/svg+xml' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `mindmap_${new Date().getTime()}.svg`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
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
  if (mmInstance) {
    mmInstance.setScale(scale.value)
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
}

.mindmap-header {
  padding: 16px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #F2F3F5;
  background-color: #FFFFFF;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
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