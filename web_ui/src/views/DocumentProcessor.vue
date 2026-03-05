<template>
  <div class="document-processor">
    <div class="workspace">
      <!-- 导航栏 -->
      <header class="nav-bar">
        <h1 class="app-title">智能解析 <span class="subtitle">让文档内容为AI所用</span></h1>
        <div class="header-actions">
          <el-button 
            v-if="isHeaderCollapsed"
            type="primary" 
            :icon="Upload"
            @click="toggleUploadArea"
            class="upload-toggle-button"
          >
            重新上传
          </el-button>
          <el-button 
            type="default" 
            :icon="Setting"
            @click="toggleSettings"
            class="settings-button"
          >
            设置
          </el-button>
        </div>
      </header>
      
      <!-- 拖拽上传区 -->
      <div 
        v-if="!isHeaderCollapsed"
        class="drag-upload-area"
        :class="{ 'drag-over': isDragging }"
        @drop="handleDrop"
        @dragover.prevent="isDragging = true"
        @dragleave="isDragging = false"
        @click="triggerUpload"
      >
        <div class="drag-upload-content" v-if="uploadedFiles.length === 0">
          <div class="file-icons">
            <el-icon class="file-icon pdf-icon"><Document /></el-icon>
            <el-icon class="file-icon word-icon"><Document /></el-icon>
            <el-icon class="file-icon docker-icon"><Document /></el-icon>
            <el-icon class="file-icon png-icon"><Picture /></el-icon>
          </div>
          <p class="drag-upload-text">支持 PDF、Word、Docker、PNG 文件格式，点击或拖拽文件至此上传</p>
        </div>
        
        <!-- 已上传文件显示在上传框内 -->
        <div v-else class="uploaded-files-in-area">
          <div class="file-card" v-for="(file, index) in uploadedFiles" :key="index">
            <div class="file-info">
              <el-icon class="file-icon"><Document /></el-icon>
              <span class="file-name">{{ file.name }}</span>
              <span class="file-size">({{ formatFileSize(file.size) }})</span>
              <el-button 
                type="danger" 
                :icon="Delete" 
                circle 
                size="small"
                @click.stop="removeFile(index)"
                class="remove-button"
              />
            </div>
          </div>
        </div>
      </div>
      

      
      <!-- 主内容区 -->
      <div class="content-container">
        
        <!-- 设置面板 -->
        <div v-if="showSettings" class="settings-panel">
          <ConfigPanel 
            v-model="config"
            :backend-options="backendOptions"
            :language-options="languageOptions"
            @backend-change="handleBackendChange"
          />
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
        
        <!-- 结果面板 -->
        <div class="result-panel-container">
          <ResultPanel :result="results" />
        </div>
      </div>
    </div>
    
    <!-- 隐藏的文件输入 -->
    <input
      ref="fileInput"
      type="file"
      multiple
      accept=".pdf,.doc,.docx,.dockerfile,Dockerfile,.png"
      style="position: absolute; width: 0; height: 0; overflow: hidden;"
      @change="handleFileInputChange"
    />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { MagicStick, Delete, Setting, Upload, Document } from '@element-plus/icons-vue'
import type { UploadFile, UploadInstance } from 'element-plus'
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
  clearAll,
  processDocument: originalProcessDocument,
  getFormulaLabel,
  getFormulaInfo
} = useDocumentProcessor()

const showSettings = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)
const isHeaderCollapsed = ref(false)

const toggleSettings = () => {
  showSettings.value = !showSettings.value
}

const toggleUploadArea = () => {
  isHeaderCollapsed.value = !isHeaderCollapsed.value
}

const handleBackendChange = (backend: string) => {
  // 可以在这里添加后端切换的额外逻辑
  console.log('Backend changed to:', backend)
}

const triggerUpload = () => {
  console.log('triggerUpload called')
  console.log('fileInput.value:', fileInput.value)
  if (fileInput.value) {
    console.log('Clicking file input')
    fileInput.value.click()
  } else {
    console.log('fileInput.value is null')
  }
}

const handleFileInputChange = (event: Event) => {
  const input = event.target as HTMLInputElement
  if (input.files) {
    const files = Array.from(input.files)
    files.forEach(file => {
      uploadedFiles.value.push(file)
    })
    // 清空input值，允许重复选择同一个文件
    input.value = ''
    // 上传后自动转换
    processDocument()
  }
}

const handleDrop = (event: DragEvent) => {
  event.preventDefault()
  isDragging.value = false
  
  if (event.dataTransfer?.files) {
    const files = Array.from(event.dataTransfer.files)
    files.forEach(file => {
      uploadedFiles.value.push(file)
    })
    // 上传后自动转换
    processDocument()
  }
}

const removeFile = (index: number) => {
  uploadedFiles.value.splice(index, 1)
  // 如果没有文件了，重新显示上传区域
  if (uploadedFiles.value.length === 0) {
    isHeaderCollapsed.value = false
  }
}

// 重写clearAll函数，确保清除后显示上传区域
const clearAllFiles = () => {
  clearAll()
  isHeaderCollapsed.value = false
}

// 重写processDocument函数，在点击转换按钮后立即折叠上传区域
const processDocument = async () => {
  // 点击转换按钮后立即折叠上传区域
  isHeaderCollapsed.value = true
  // 调用useDocumentProcessor中的processDocument函数
  await originalProcessDocument()
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
</script>

<style scoped>
.document-processor {
  height: 100vh;
  overflow: hidden;
  background-color: #F2F3F5;
}

.workspace {
  height: 100%;
  background-color: #FFFFFF;
  display: flex;
  flex-direction: column;
}

/* 导航栏 */
.nav-bar {
  background-color: #ffffff;
  padding: 16px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.app-title {
  font-size: 20px;
  font-weight: 600;
  color: #1D2129;
  margin: 0;
}

.subtitle {
  font-size: 14px;
  font-weight: 400;
  color: #4E5969;
  margin-left: 12px;
}

.header-actions {
  display: flex;
  gap: 16px;
  align-items: center;
}

/* 上传按钮 */
.upload-button {
  background-color: #165DFF;
  border-color: #165DFF;
  color: #FFFFFF;
  font-weight: 500;
  border-radius: 6px;
  height: 40px;
  padding: 0 20px;
  transition: all 0.3s ease;
}

/* 重新上传按钮 */
.upload-toggle-button {
  background-color: #165DFF;
  border-color: #165DFF;
  color: #FFFFFF;
  font-weight: 500;
  border-radius: 6px;
  height: 40px;
  padding: 0 16px;
  transition: all 0.3s ease;
  margin-right: 16px;
}

.upload-button:hover {
  background-color: #0E42D2;
  border-color: #0E42D2;
}

.upload-button:active {
  background-color: #0E42D2;
  border-color: #0E42D2;
  transform: translateY(1px);
}

/* 设置按钮 */
.settings-button {
  border-color: #C9CDD4;
  color: #4E5969;
  border-radius: 6px;
  height: 40px;
  padding: 0 16px;
  transition: all 0.3s ease;
  font-size: 14px;
  font-weight: 400;
}

.settings-button:hover {
  border-color: #165DFF;
  color: #165DFF;
}

.settings-button:active {
  border-color: #0E42D2;
  color: #0E42D2;
  transform: translateY(1px);
}

/* 拖拽上传区 */
.drag-upload-area {
  border: 1px dashed #DCDFE6;
  background-color: #F9FAFC;
  padding: 48px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  overflow: hidden;
  max-height: 200px;
}

/* 已上传文件容器 */
.uploaded-files-container {
  padding: 24px;
  background-color: #FFFFFF;
  border-bottom: 1px solid #EEEEEE;
  transition: all 0.3s ease;
}

/* 已上传文件在上传框内显示 */
.uploaded-files-in-area {
  width: 100%;
}

.uploaded-files-in-area .file-card {
  background-color: #F9FAFC;
  border: 1px solid #E4E7ED;
  border-radius: 6px;
  padding: 16px;
  margin-bottom: 16px;
  transition: all 0.3s ease;
}

.uploaded-files-in-area .file-card:hover {
  border-color: #165DFF;
  box-shadow: 0 2px 8px rgba(22, 93, 255, 0.1);
}



.drag-upload-area:hover {
  border-color: #165DFF;
  background-color: #ECF5FF;
}

.drag-upload-area.drag-over {
  border-color: #165DFF;
  background-color: #ECF5FF;
}

.drag-upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

.file-icons {
  display: flex;
  align-items: center;
  gap: 16px;
  position: relative;
  height: 64px;
}

.file-icon {
  font-size: 32px;
  position: absolute;
  transition: all 0.3s ease;
}

.pdf-icon {
  color: #FF0000;
  left: 0;
  z-index: 4;
  transform: translateX(0);
}

.word-icon {
  color: #1E40AF;
  left: 24px;
  z-index: 3;
  transform: translateX(10%);
}

.docker-icon {
  color: #2496ED;
  left: 48px;
  z-index: 2;
  transform: translateX(20%);
}

.png-icon {
  color: #FF6B6B;
  left: 72px;
  z-index: 1;
  transform: translateX(30%);
}

.drag-upload-text {
  font-size: 14px;
  color: #4E5969;
  margin: 0;
  max-width: 600px;
  line-height: 1.5;
}

/* 主内容区 */
.content-container {
  flex: 1;
  overflow: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  min-height: 0;
}

/* 操作按钮 */
.action-buttons {
  display: flex;
  gap: 16px;
  width: 100%;
}

.action-button {
  flex: 1;
  height: 40px;
  font-size: 14px;
  border-radius: 6px;
  transition: all 0.3s ease;
  padding: 0 20px;
}

.primary-button {
  background-color: #165DFF;
  border-color: #165DFF;
  color: #FFFFFF;
  font-weight: 500;
}

.primary-button:hover {
  background-color: #0E42D2;
  border-color: #0E42D2;
}

.primary-button:active {
  background-color: #0E42D2;
  border-color: #0E42D2;
  transform: translateY(1px);
}

.secondary-button {
  background-color: #FFFFFF;
  border-color: #C9CDD4;
  color: #4E5969;
  font-weight: 400;
}

.secondary-button:hover {
  border-color: #165DFF;
  color: #165DFF;
}

.secondary-button:active {
  border-color: #0E42D2;
  color: #0E42D2;
  transform: translateY(1px);
}

/* 文件上传提示 */
.upload-tip {
  padding: 24px;
  background-color: #F2F3F5;
  border-radius: 8px;
  text-align: center;
  margin-bottom: 16px;
}

.tip-text {
  font-size: 14px;
  color: #4E5969;
  margin: 0;
}

/* 已上传文件 */
.uploaded-files {
  width: 100%;
}

.files-title {
  font-size: 16px;
  font-weight: 500;
  color: #1D2129;
  margin: 0 0 16px 0;
}

.file-card {
  background-color: #F2F3F5;
  border-radius: 6px;
  padding: 16px;
  margin-bottom: 12px;
  transition: all 0.3s ease;
}

.file-card:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.file-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.file-icon {
  color: #165DFF;
  font-size: 18px;
}

.file-name {
  flex: 1;
  font-weight: 400;
  color: #4E5969;
  font-size: 14px;
}

.file-size {
  color: #4E5969;
  font-size: 12px;
}

.remove-button {
  opacity: 0.7;
  transition: opacity 0.3s ease;
}

.remove-button:hover {
  opacity: 1;
}

/* 设置面板 */
.settings-panel {
  background-color: #F2F3F5;
  border-radius: 8px;
  padding: 24px;
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

.error-alert {
  border-radius: 8px;
}

/* 结果面板 */
.result-panel-container {
  flex: 1;
  min-height: 0;
  margin-top: 16px;
}

/* 隐藏的上传控件 */
.hidden-upload {
  display: none;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .nav-bar {
    padding: 12px 16px;
  }
  
  .app-title {
    font-size: 18px;
  }
  
  .subtitle {
    font-size: 12px;
  }
  
  .header-actions {
    gap: 8px;
  }
  
  .upload-button,
  .settings-button {
    padding: 0 12px;
    font-size: 12px;
  }
  
  .content-container {
    padding: 16px;
    gap: 16px;
  }
  
  .action-buttons {
    flex-direction: column;
  }
  
  .action-button {
    width: 100%;
  }
}
</style>