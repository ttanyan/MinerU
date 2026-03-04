<template>
  <div class="result-panel">
    <div class="panel-header">
      <span class="panel-title">{{ $t('results.title') }}</span>
      <el-button 
        v-if="result" 
        type="primary" 
        :icon="Download"
        @click="downloadResult"
        size="small"
      >
        {{ $t('results.download') }}
      </el-button>
    </div>
    
    <el-tabs v-model="activeTab" class="result-tabs" type="card" :border="false">
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
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.panel-title {
  font-weight: 500;
  font-size: 16px;
  color: #303133;
}

.result-tabs {
  flex: 1;
  min-height: 0;
}

.result-tabs :deep(.el-tabs__content) {
  height: calc(100% - 40px);
  padding-top: 20px;
}

.result-tabs :deep(.el-tab-pane) {
  height: 100%;
}

.result-tabs :deep(.el-tabs__nav) {
  border-bottom: 1px solid #E4E7ED;
}

.result-tabs :deep(.el-tabs__item) {
  height: 40px;
  line-height: 40px;
  padding: 0 20px;
  margin-right: 40px;
  color: #606266;
  font-size: 14px;
  position: relative;
}

.result-tabs :deep(.el-tabs__item.is-active) {
  color: #1677FF;
  font-weight: 500;
}

.result-tabs :deep(.el-tabs__item.is-active::after) {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 2px;
  background-color: #1677FF;
  border-radius: 1px;
}

.result-tabs :deep(.el-tabs__item:hover) {
  color: #1677FF;
}

.source-textarea {
  height: 100%;
}

.source-textarea :deep(.el-textarea__inner) {
  height: 100%;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 14px;
  line-height: 1.5;
  border: 1px solid #E4E7ED;
  border-radius: 4px;
  color: #303133;
}
</style>