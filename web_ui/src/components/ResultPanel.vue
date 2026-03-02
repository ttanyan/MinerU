<template>
  <el-card class="result-panel">
    <template #header>
      <div class="card-header">
        <span>{{ $t('results.title') }}</span>
        <el-button 
          v-if="result" 
          type="primary" 
          icon="download"
          @click="downloadResult"
        >
          {{ $t('results.download') }}
        </el-button>
      </div>
    </template>
    
    <el-tabs v-model="activeTab" class="result-tabs">
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
  </el-card>
</template>

<script setup lang="ts">
import { ref } from 'vue'
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
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.result-tabs {
  height: calc(100% - 60px);
}

.result-tabs :deep(.el-tabs__content) {
  height: calc(100% - 55px);
}

.result-tabs :deep(.el-tab-pane) {
  height: 100%;
}

.source-textarea {
  height: 100%;
}

.source-textarea :deep(.el-textarea__inner) {
  height: 100%;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 14px;
  line-height: 1.5;
}
</style>