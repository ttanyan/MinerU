<template>
  <div class="document-processor">
    <!-- 顶部标题栏 -->
    <header class="top-header">
      <h1 class="page-title">思维导图</h1>
      <el-button 
        type="text" 
        @click="toggleSettings"
        class="settings-button"
      >
        设置
      </el-button>
    </header>
    
    <!-- 主内容区域 -->
    <div class="main-content">
      <!-- 文件上传区域 -->
      <div class="upload-section">
        <!-- 拖拽上传区 -->
        <div 
          class="drag-upload-area"
          :class="{ 'drag-over': isDragging, 'collapsed': isUploadAreaCollapsed }"
          @drop="handleDrop"
          @dragover.prevent="isDragging = true"
          @dragleave="isDragging = false"
          @click="triggerUpload"
        >
          <div class="upload-content" v-show="!isUploadAreaCollapsed">
            <el-icon class="upload-icon"><Folder /></el-icon>
            <div class="upload-text">文件导入</div>
            <div class="upload-hint">支持PDF、Word、PNG格式文件，点击或拖拽文件至此处上传</div>
          </div>
          
          <!-- 已上传文件列表 -->
          <div class="uploaded-files">
            <div class="file-item" v-for="(file, index) in uploadedFiles" :key="index">
              <el-icon class="file-icon"><Document /></el-icon>
              <span class="file-name">{{ file.name }}</span>
              <el-button 
                type="text" 
                :icon="Delete" 
                @click.stop="removeFile(index)"
                class="remove-button"
              />
            </div>
          </div>
        </div>
        

        
        <!-- 设置面板 -->
        <div v-if="showSettings" class="settings-panel">
          <div class="settings-header">
            <h3 class="settings-title">设置</h3>
            <el-button 
              type="text" 
              @click="toggleSettings"
              class="close-button"
            >
              <el-icon><Close /></el-icon>
            </el-button>
          </div>
          <ConfigPanel 
            v-model="config"
            :backend-options="backendOptions"
            :language-options="languageOptions"
            @backend-change="handleBackendChange"
          />
        </div>
      </div>
      
      <!-- 结果显示区域 -->
      <div class="result-section">
        <!-- 标签页 -->
        <el-tabs v-model="activeTab" class="result-tabs">
          <el-tab-pane label="Markdown 渲染" name="markdown" />
          <el-tab-pane label="Markdown 源码" name="source" />
          <el-tab-pane label="思维导图" name="mindmap" />
        </el-tabs>
        
        <!-- 结果内容 -->
        <div class="result-content">
          <!-- 加载状态 -->
          <div v-if="isProcessing" class="loading-container">
            <el-spinner :size="48" />
            <p class="loading-text">正在处理文档...</p>
          </div>
          
          <!-- 无结果状态 -->
          <div v-else-if="!results" class="empty-state">
            <div class="empty-icon"></div>
            <p class="empty-text">暂无转换结果</p>
          </div>
          
          <!-- 结果内容 -->
          <div v-else class="result-tab-content">
            <!-- Markdown 渲染 -->
            <div v-show="activeTab === 'markdown'" class="markdown-content">
              <MarkdownRenderer :content="results.markdown || ''" />
            </div>
            
            <!-- Markdown 源码 -->
            <div v-show="activeTab === 'source'" class="source-content">
              <el-input
                :model-value="results.source"
                type="textarea"
                :rows="20"
                readonly
                class="source-textarea"
              />
            </div>
            
            <!-- 思维导图 -->
            <div v-show="activeTab === 'mindmap'" class="mindmap-content">
              <MindMapRenderer :content="results.mindmap || ''" />
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 隐藏的文件输入 -->
    <input
      ref="fileInput"
      type="file"
      multiple
      accept=".pdf,.doc,.docx,.png"
      style="position: absolute; width: 0; height: 0; overflow: hidden;"
      @change="handleFileInputChange"
    />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Delete, Document, Folder, Setting, Close } from '@element-plus/icons-vue'
import ConfigPanel from '@/components/ConfigPanel.vue'
import MarkdownRenderer from '@/components/MarkdownRenderer.vue'
import MindMapRenderer from '@/components/MindMapRenderer.vue'
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
  processDocument: originalProcessDocument
} = useDocumentProcessor()

const showSettings = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)
const activeTab = ref('markdown')
const isUploadAreaCollapsed = ref(false)

const toggleSettings = () => {
  showSettings.value = !showSettings.value
}

const handleBackendChange = (backend: string) => {
  console.log('Backend changed to:', backend)
}

const triggerUpload = () => {
  if (fileInput.value) {
    fileInput.value.click()
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
    // 折叠上传区域
    isUploadAreaCollapsed.value = true
    // 自动处理文件
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
    // 折叠上传区域
    isUploadAreaCollapsed.value = true
    // 自动处理文件
    processDocument()
  }
}

const removeFile = (index: number) => {
  uploadedFiles.value.splice(index, 1)
  // 如果没有文件了，展开上传区域
  if (uploadedFiles.value.length === 0) {
    isUploadAreaCollapsed.value = false
  }
}

const processDocument = async () => {
  await originalProcessDocument()
  // 处理完成后切换到思维导图标签
  activeTab.value = 'mindmap'
}

// 重写clearAll函数，确保清除后显示上传区域
const clearAllFiles = () => {
  clearAll()
  isUploadAreaCollapsed.value = false
}
</script>

<style scoped>
.document-processor {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #F8F9FA;
}

/* 顶部标题栏 */
.top-header {
  padding: 16px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #FFFFFF;
  border-bottom: 1px solid #E9ECEF;
}

.page-title {
  font-size: 18px;
  font-weight: 500;
  color: #343A40;
  margin: 0;
}

.settings-button {
  color: #6C757D;
  font-size: 14px;
  padding: 4px 12px;
}

.settings-button:hover {
  color: #165DFF;
}

/* 主内容区域 */
.main-content {
  flex: 1;
  padding: 24px;
  overflow: auto;
}

/* 上传区域 */
.upload-section {
  margin-bottom: 32px;
}

.section-title {
  font-size: 16px;
  font-weight: 500;
  color: #343A40;
  margin: 0 0 16px 0;
}

/* 拖拽上传区 */
.drag-upload-area {
  border: 2px dashed #CED4DA;
  border-radius: 8px;
  padding: 40px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background-color: #FFFFFF;
  margin-bottom: 16px;
  overflow: hidden;
}

.drag-upload-area.collapsed {
  padding: 16px 24px;
  min-height: 60px;
  max-height: 100px;
}

.drag-upload-area:hover {
  border-color: #165DFF;
  background-color: #F8F9FF;
}

.drag-upload-area.drag-over {
  border-color: #165DFF;
  background-color: #F8F9FF;
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.upload-icon {
  font-size: 48px;
  color: #165DFF;
}

.upload-text {
  font-size: 16px;
  font-weight: 500;
  color: #343A40;
}

.upload-hint {
  font-size: 14px;
  color: #6C757D;
  line-height: 1.5;
  max-width: 400px;
}

/* 已上传文件 */
.uploaded-files {
  margin-top: 24px;
  text-align: left;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background-color: #F8F9FA;
  border-radius: 6px;
  margin-bottom: 8px;
  transition: all 0.3s ease;
}

.file-item:hover {
  background-color: #E9ECEF;
}

.file-icon {
  font-size: 18px;
  color: #165DFF;
}

.file-name {
  flex: 1;
  font-size: 14px;
  color: #343A40;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.remove-button {
  color: #6C757D;
  padding: 4px;
}

.remove-button:hover {
  color: #DC3545;
}

/* 操作按钮 */
.action-buttons {
  display: flex;
  gap: 12px;
  margin-top: 8px;
}

.process-button {
  flex: 1;
  background-color: #165DFF;
  border-color: #165DFF;
  color: #FFFFFF;
  border-radius: 6px;
  height: 40px;
  font-size: 14px;
  transition: all 0.3s ease;
}

.process-button:hover:not(:disabled) {
  background-color: #0E42D2;
  border-color: #0E42D2;
}

.clear-button {
  flex: 1;
  background-color: #FFFFFF;
  border-color: #CED4DA;
  color: #495057;
  border-radius: 6px;
  height: 40px;
  font-size: 14px;
  transition: all 0.3s ease;
}

.clear-button:hover:not(:disabled) {
  border-color: #165DFF;
  color: #165DFF;
}

/* 设置面板 */
.settings-panel {
  position: fixed;
  top: 0;
  right: 0;
  width: 320px;
  height: 100vh;
  background-color: #FFFFFF;
  border-left: 1px solid #E9ECEF;
  box-shadow: -4px 0 12px rgba(0, 0, 0, 0.1);
  padding: 24px;
  z-index: 1000;
  animation: slideInFromRight 0.3s ease;
  overflow-y: auto;
}

.settings-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid #E9ECEF;
}

.close-button {
  color: #6C757D;
  padding: 4px;
  font-size: 16px;
}

.close-button:hover {
  color: #343A40;
}

.settings-title {
  font-size: 16px;
  font-weight: 500;
  color: #343A40;
  margin: 0;
}

@keyframes slideInFromRight {
  from {
    opacity: 0;
    transform: translateX(100%);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

/* 结果区域 */
.result-section {
  background-color: #FFFFFF;
  border: 1px solid #E9ECEF;
  border-radius: 8px;
  overflow: hidden;
}

/* 标签页 */
.result-tabs {
  border-bottom: 1px solid #E9ECEF;
}

.result-tabs :deep(.el-tabs__nav) {
  padding-left: 24px;
}

.result-tabs :deep(.el-tabs__item) {
  height: 48px;
  line-height: 48px;
  padding: 0 24px;
  margin-right: 0;
  color: #6C757D;
  font-size: 14px;
  font-weight: 400;
  transition: all 0.3s ease;
}

.result-tabs :deep(.el-tabs__item:hover) {
  color: #165DFF;
}

.result-tabs :deep(.el-tabs__item.is-active) {
  color: #165DFF;
  font-weight: 500;
}

.result-tabs :deep(.el-tabs__active-bar) {
  background-color: #165DFF;
  height: 2px;
}

/* 结果内容 */
.result-content {
  min-height: 400px;
  padding: 24px;
}

/* 加载状态 */
.loading-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 400px;
  gap: 16px;
}

.loading-text {
  font-size: 14px;
  color: #6C757D;
  margin: 0;
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 400px;
  gap: 16px;
}

.empty-icon {
  width: 120px;
  height: 120px;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Crect x='20' y='30' width='80' height='60' rx='4' fill='%23F8F9FA' stroke='%23E9ECEF' stroke-width='2'/%3E%3Crect x='40' y='15' width='40' height='20' rx='2' fill='%23F8F9FA' stroke='%23E9ECEF' stroke-width='2'/%3E%3Crect x='30' y='40' width='60' height='10' rx='2' fill='%23E9ECEF'/%3E%3Crect x='30' y='55' width='50' height='8' rx='2' fill='%23E9ECEF'/%3E%3Crect x='30' y='70' width='40' height='8' rx='2' fill='%23E9ECEF'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: center;
}

.empty-text {
  font-size: 14px;
  color: #6C757D;
  margin: 0;
}

/* 标签内容 */
.result-tab-content {
  min-height: 400px;
}

.markdown-content {
  min-height: 400px;
  line-height: 1.6;
  color: #343A40;
}

.source-content {
  min-height: 400px;
}

.source-textarea {
  height: 100%;
  min-height: 400px;
}

.source-textarea :deep(.el-textarea__wrapper) {
  border: 1px solid #E9ECEF;
  border-radius: 4px;
  box-shadow: none;
  height: 100%;
  min-height: 400px;
}

.source-textarea :deep(.el-textarea__inner) {
  height: 100%;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 14px;
  line-height: 1.5;
  color: #495057;
  padding: 16px;
}

.mindmap-content {
  min-height: 400px;
  height: 100%;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .main-content {
    padding: 16px;
  }
  
  .top-header {
    padding: 12px 16px;
  }
  
  .page-title {
    font-size: 16px;
  }
  
  .drag-upload-area {
    padding: 32px 16px;
  }
  
  .upload-icon {
    font-size: 32px;
  }
  
  .upload-text {
    font-size: 14px;
  }
  
  .upload-hint {
    font-size: 12px;
  }
  
  .action-buttons {
    flex-direction: column;
  }
  
  .result-content {
    padding: 16px;
  }
  
  .result-tabs :deep(.el-tabs__nav) {
    padding-left: 16px;
  }
  
  .result-tabs :deep(.el-tabs__item) {
    padding: 0 16px;
    font-size: 12px;
  }
}
</style>