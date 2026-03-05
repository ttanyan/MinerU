<template>
  <div class="file-uploader">
    <div 
      class="upload-container"
      :class="{ 'drag-over': isDragging }"
      @drop="handleDrop"
      @dragover.prevent="isDragging = true"
      @dragleave="isDragging = false"
      @click="triggerUpload"
    >
      <div class="upload-content">
        <el-icon class="upload-icon"><UploadFilled /></el-icon>
        <div class="upload-text">
          <div class="main-text">点击或拖拽文件到此处上传</div>
          <div class="sub-text">支持 PDF、PNG、JPG、JPEG 格式，最大 100MB</div>
        </div>
      </div>
    </div>
    
    <!-- 隐藏的上传控件 -->
    <el-upload
      ref="uploadRef"
      :auto-upload="false"
      :show-file-list="false"
      :on-change="handleFileChange"
      :before-upload="beforeUpload"
      multiple
      accept=".pdf,.png,.jpg,.jpeg"
      class="hidden-upload"
    >
      <el-button type="primary">上传文件</el-button>
    </el-upload>
    
    <!-- 已上传文件预览 -->
    <div v-if="files.length > 0" class="file-preview">
      <h3 class="preview-title">已上传文件</h3>
      <div class="file-card" v-for="(file, index) in files" :key="index">
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
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { UploadFilled, Document, Delete, Picture } from '@element-plus/icons-vue'
import type { UploadFile, UploadInstance } from 'element-plus'

interface Props {
  modelValue: File[]
}

interface Emits {
  (e: 'update:modelValue', value: File[]): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const files = ref<File[]>(props.modelValue)
const uploadRef = ref<UploadInstance>()
const isDragging = ref(false)

// 监听外部值变化
watch(() => props.modelValue, (newVal) => {
  files.value = newVal
})

// 监听内部值变化并通知父组件
watch(files, (newVal) => {
  emit('update:modelValue', newVal)
}, { deep: true })

const triggerUpload = () => {
  uploadRef.value?.handleClick()
}

const handleFileChange = (uploadFile: UploadFile) => {
  if (uploadFile.raw) {
    files.value.push(uploadFile.raw)
  }
}

const handleDrop = (event: DragEvent) => {
  event.preventDefault()
  isDragging.value = false
  
  if (event.dataTransfer?.files) {
    const droppedFiles = Array.from(event.dataTransfer.files)
    droppedFiles.forEach(file => {
      if (file.type.match(/(pdf|png|jpe?g)/i)) {
        files.value.push(file)
      }
    })
  }
}

const beforeUpload = (file: File): boolean => {
  // 阻止自动上传
  return false
}

const removeFile = (index: number) => {
  files.value.splice(index, 1)
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
.file-uploader {
  margin-bottom: 24px;
}

.upload-container {
  border: 2px dashed #C9CDD4;
  border-radius: 8px;
  background-color: #FFFFFF;
  padding: 60px 24px;
  text-align: center;
  transition: all 0.3s ease;
  margin-bottom: 24px;
  cursor: pointer;
}

.upload-container:hover {
  border-color: #165DFF;
  background-color: #E8F3FF;
}

.upload-container.drag-over {
  border-color: #165DFF;
  background-color: #E8F3FF;
}

.upload-container.drag-over .upload-icon {
  transform: scale(1.1);
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

.upload-icon {
  font-size: 24px;
  color: #165DFF;
  transition: transform 0.3s ease;
}

.upload-text {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.main-text {
  font-size: 16px;
  font-weight: 400;
  color: #1D2129;
}

.sub-text {
  font-size: 12px;
  color: #4E5969;
}

.hidden-upload {
  display: none;
}

.file-preview {
  margin-top: 20px;
}

.preview-title {
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
</style>