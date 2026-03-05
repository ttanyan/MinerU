<template>
  <div class="result-panel">
    <div class="panel-header" v-if="result">
      <el-button 
        type="primary" 
        :icon="Download"
        @click="downloadResult"
        size="small"
        class="download-button"
      >
        {{ $t('results.download') }}
      </el-button>
    </div>
    
    <!-- 加载状态 -->
    <div v-if="isProcessing" class="loading-container">
      <el-spinner :size="64" />
      <p class="loading-text">处理中...</p>
    </div>
    
    <!-- 结果内容 -->
    <el-tabs v-else v-model="activeTab" class="result-tabs">
      <el-tab-pane :label="$t('results.tabs.markdown')" name="markdown">
        <MarkdownRenderer :content="result?.markdown || ''" />
      </el-tab-pane>
      
      <el-tab-pane :label="$t('results.tabs.source')" name="source">
        <el-input
          :model-value="result?.source"
          type="textarea"
          :rows="20"
          readonly
          class="source-textarea"
        />
      </el-tab-pane>
      
      <el-tab-pane :label="$t('results.tabs.mindmap')" name="mindmap">
        <MindMapRenderer :content="result?.mindmap || ''" />
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Download } from '@element-plus/icons-vue'
import type { ProcessResult } from '@/composables/useDocumentProcessor'
import MarkdownRenderer from './MarkdownRenderer.vue'
import MindMapRenderer from './MindMapRenderer.vue'

interface Props {
  result: ProcessResult | null
  isProcessing: boolean
}

const props = defineProps<Props>()

const activeTab = ref('markdown')

const downloadResult = () => {
  if (!props.result?.markdown) return
  
  const blob = new Blob([props.result.markdown], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `result_${new Date().getTime()}.md`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
</script>

<style scoped>
.result-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.panel-header {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  margin-bottom: 16px;
}

.download-button {
  border-radius: 6px;
  transition: all 0.3s ease;
  background-color: #165DFF;
  border-color: #165DFF;
}

.download-button:hover {
  background-color: #0E42D2;
  border-color: #0E42D2;
  transform: scale(1.05);
}

.result-tabs {
  flex: 1;
  min-height: 0;
}

.result-tabs :deep(.el-tabs__content) {
  height: calc(100% - 48px);
  padding-top: 16px;
}

.result-tabs :deep(.el-tab-pane) {
  height: 100%;
}

/* 极简Tab样式 */
.result-tabs :deep(.el-tabs__nav) {
  border: none;
  padding: 0;
}

.result-tabs :deep(.el-tabs__item) {
  height: 48px;
  line-height: 48px;
  padding: 0 24px;
  margin-right: 24px;
  color: #4E5969;
  font-size: 14px;
  font-weight: 400;
  position: relative;
  transition: all 0.3s ease;
}

.result-tabs :deep(.el-tabs__item:hover) {
  color: #165DFF;
}

.result-tabs :deep(.el-tabs__item.is-active) {
  color: #165DFF;
  font-weight: 500;
}

.result-tabs :deep(.el-tabs__item.is-active::after) {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 2px;
  background-color: #165DFF;
  border-radius: 1px;
  transition: all 0.3s ease;
}

.result-tabs :deep(.el-tabs__active-bar) {
  display: none;
}

.source-textarea {
  height: 100%;
}

.source-textarea :deep(.el-textarea__wrapper) {
  border: 1px solid #C9CDD4;
  border-radius: 8px;
  box-shadow: none;
  transition: all 0.3s ease;
}

.source-textarea :deep(.el-textarea__wrapper:hover) {
  border-color: #165DFF;
}

.source-textarea :deep(.el-textarea__inner) {
  height: 100%;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 14px;
  line-height: 1.5;
  color: #4E5969;
  padding: 16px;
}

/* 加载状态 */
.loading-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-height: 400px;
}

.loading-text {
  margin-top: 16px;
  font-size: 16px;
  color: #4E5969;
}
</style>