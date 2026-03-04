<template>
  <div class="document-processor">
    <div class="header-section">
      <h1 class="app-title">智能解析 <span class="subtitle">让文档内容为AI所用</span></h1>
    </div>
    
    <el-row :gutter="30" class="main-content">
      <!-- 左侧上传和设置区域 -->
      <el-col :span="12">
        <div class="upload-card">
          <div class="upload-header">
            <h2 class="card-title">上传文件</h2>
            <el-button 
              type="default" 
              :icon="Setting"
              size="small"
              @click="toggleSettings"
              class="settings-button"
            >
              设置
            </el-button>
          </div>
          
          <!-- 文件上传 -->
          <FileUploader 
            v-model="uploadedFiles"
            class="upload-section"
          />
          
          <!-- 设置面板 -->
          <div v-if="showSettings" class="settings-panel">
            <ConfigPanel 
              v-model="config"
              :backend-options="backendOptions"
              :language-options="languageOptions"
              @backend-change="handleBackendChange"
            />
          </div>
          
          <!-- 操作按钮 -->
          <div class="action-buttons">
            <el-button 
              type="primary" 
              size="large"
              :loading="isProcessing"
              @click="processDocument"
              :disabled="uploadedFiles.length === 0"
              class="convert-button"
            >
              <el-icon><MagicStick /></el-icon>
              {{ $t('common.convert') }}
            </el-button>
            
            <el-button 
              @click="clearAll"
              size="large"
              class="clear-button"
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
        </div>
      </el-col>
      
      <!-- 右侧结果面板 -->
      <el-col :span="12">
        <div class="result-card">
          <h2 class="card-title">处理结果</h2>
          <ResultPanel :result="results" />
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { MagicStick, Delete, Setting } from '@element-plus/icons-vue'
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

const showSettings = ref(false)

const toggleSettings = () => {
  showSettings.value = !showSettings.value
}

const handleBackendChange = (backend: string) => {
  // 可以在这里添加后端切换的额外逻辑
  console.log('Backend changed to:', backend)
}
</script>

<style scoped>
.document-processor {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.header-section {
  margin-bottom: 30px;
}

.app-title {
  font-size: 20px;
  font-weight: 600;
  color: #409EFF;
  margin: 0;
}

.subtitle {
  font-size: 14px;
  font-weight: 400;
  color: #606266;
  margin-left: 10px;
}

.main-content {
  flex: 1;
  min-height: 0;
}

.upload-card,
.result-card {
  background-color: #ffffff;
  border: 1px solid #E4E7ED;
  border-radius: 8px;
  padding: 30px;
  height: 100%;
  box-sizing: border-box;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
}

.upload-card:hover,
.result-card:hover {
  box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.12);
}

.upload-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.card-title {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
  margin: 0;
}

.settings-button {
  border-color: #DCDFE6;
  color: #606266;
}

.settings-button:hover {
  border-color: #1677FF;
  color: #1677FF;
}

.upload-section {
  margin-bottom: 24px;
}

.settings-panel {
  background-color: #F9FAFC;
  border: 1px solid #E4E7ED;
  border-radius: 6px;
  padding: 20px;
  margin-bottom: 24px;
  animation: slideDown 0.3s ease;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.action-buttons {
  display: flex;
  gap: 16px;
  margin: 24px 0;
}

.convert-button,
.clear-button {
  flex: 1;
  height: 48px;
  font-size: 16px;
}

.error-alert {
  margin-top: 24px;
}

@media (max-width: 1200px) {
  .main-content {
    flex-direction: column;
  }
  
  .el-col {
    width: 100% !important;
    margin-bottom: 30px;
  }
  
  .upload-card,
  .result-card {
    height: auto;
  }
}
</style>