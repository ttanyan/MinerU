<template>
  <div class="mindmap-container">
    <svg ref="svgRef" class="markmap-svg"></svg>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, onUnmounted, nextTick } from 'vue'
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

const initMarkmap = async () => {
  if (!props.content || !svgRef.value) return

  try {
    // 1. 转换数据 (对应原代码中的 transformer.transform)
    const { root, features } = transformer.transform(props.content)

    // 2. 加载必要的资源 (对应原代码中的 loadCSS/loadJS)
    const { styles, scripts } = transformer.getAssets()
    if (styles) loadCSS(styles)
    if (scripts) loadJS(scripts)

    // 3. 创建或更新实例
    // Vue 的 nextTick 确保 DOM 已完全渲染，替代了原代码中的 setTimeout 500ms
    await nextTick()

    if (mmInstance) {
      mmInstance.setData(root)
      mmInstance.fit()
    } else {
      mmInstance = Markmap.create(svgRef.value, {
        autoFit: true,
        fitRatio: 0.9,      // 对应原代码配置
        initialExpandLevel: -1 // 对应原代码配置
      }, root)
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
  window.addEventListener('resize', () => mmInstance?.fit())
})

onUnmounted(() => {
  window.removeEventListener('resize', () => mmInstance?.fit())
  if (mmInstance) {
    mmInstance.destroy()
  }
})
</script>

<style scoped>
/* 对应原 HTML 中的 style 部分 */
.mindmap-container {
  width: 100%;
  height: 800px; /* 或者使用 100% 取决于父容器 */
  margin: 0;
  padding: 0;
  overflow: hidden;
  background-color: white;
  border: 1px solid #ddd;
  border-radius: 8px;
}

.markmap-svg {
  width: 100%;
  height: 100%;
}
</style>