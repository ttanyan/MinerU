<template>
  <div class="document-processor">
    <el-row :gutter="20" class="main-row">
      <!-- 左侧控制面板 -->
      <el-col :span="12">
        <!-- 文件上传 -->
        <FileUploader 
          v-model="uploadedFiles"
          class="upload-section"
        />
        
        <!-- 配置面板 -->
        <ConfigPanel 
          v-model="config"
          :backend-options="backendOptions"
          :language-options="languageOptions"
          @backend-change="handleBackendChange"
        />
        
        <!-- 操作按钮 -->
        <div class="action-buttons">
          <el-button 
            type="primary" 
            size="large"
            :loading="isProcessing"
            @click="processDocument"
            :disabled="uploadedFiles.length === 0"
          >
            <el-icon><MagicStick /></el-icon>
            {{ $t('common.convert') }}
          </el-button>
          
          <el-button 
            @click="clearAll"
            size="large"
          >
            <el-icon><Delete /></el-icon>
            {{ $t('common.clear') }}
          </el-button>
        </div>
        
        <!-- 错误提示 -->
        <el-alert
          v-if="error"
          :title="error"
          type="error"
          show-icon
          closable
          @close="error = null"
          class="error-alert"
        />
      </el-col>
      
      <!-- 右侧结果面板 -->
      <el-col :span="12">
        <ResultPanel :result="results" />
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { MagicStick, Delete } from '@element-plus/icons-vue'
import FileUploader from '@/components/FileUploader.vue'
import ConfigPanel from '@/components/ConfigPanel.vue'
import ResultPanel from '@/components/ResultPanel.vue'
import { useDocumentProcessor } from '@/composables/useDocumentProcessor'

// 使用组合式函数
const {
  uploadedFiles,
  config,
  results,
  isProcessing,
  error,
  backendOptions,
  languageOptions,
  handleFileUpload,
  clearAll,
  processDocument,
  getFormulaLabel,
  getFormulaInfo
} = useDocumentProcessor()

const handleBackendChange = (backend: string) => {
  // 可以在这里添加后端切换的额外逻辑
  console.log('Backend changed to:', backend)
}
</script>

<style scoped>
.document-processor {
  height: calc(100vh - 100px);
}

.main-row {
  height: 100%;
}

.upload-section {
  margin-bottom: 20px;
}

.action-buttons {
  display: flex;
  gap: 15px;
  margin: 20px 0;
}

.action-buttons .el-button {
  flex: 1;
}

.error-alert {
  margin-top: 20px;
}

@media (max-width: 1200px) {
  .main-row {
    flex-direction: column;
  }
  
  .el-col {
    width: 100% !important;
    margin-bottom: 20px;
  }
}
</style>