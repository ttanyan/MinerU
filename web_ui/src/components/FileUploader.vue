<template>
  <div class="file-uploader">
    <div class="upload-container">
      <div class="upload-content">
        <div class="upload-icon-container">
          <el-icon class="main-upload-icon"><UploadFilled /></el-icon>
        </div>
        <div class="upload-buttons">
          <el-button 
            type="primary" 
            @click="triggerUpload"
            class="upload-button"
          >
            上传文件
          </el-button>
        </div>
        <div class="upload-tip">
          支持 PDF、PNG、JPG、JPEG 格式，最大 100MB
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
            @click="removeFile(index)"
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
  border: 1px dashed #DCDFE6;
  border-radius: 8px;
  background-color: #F9FAFC;
  padding: 40px 20px;
  text-align: center;
  transition: all 0.3s ease;
  margin-bottom: 24px;
}

.upload-container:hover {
  border-color: #1677FF;
  background-color: #ECF5FF;
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
}

.upload-icon-container {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background-color: #ECF5FF;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 8px;
}

.main-upload-icon {
  font-size: 40px;
  color: #1677FF;
  opacity: 0.9;
}

.upload-buttons {
  display: flex;
  gap: 12px;
}

.upload-button {
  padding: 12px 32px;
  font-size: 16px;
  border-radius: 6px;
  background-color: #1677FF;
  border-color: #1677FF;
  transition: all 0.3s ease;
}

.upload-button:hover {
  background-color: #409EFF;
  border-color: #409EFF;
}

.upload-button:active {
  background-color: #096DD9;
  border-color: #096DD9;
}

.upload-tip {
  font-size: 14px;
  color: #909399;
  margin-top: 10px;
}

.hidden-upload {
  display: none;
}

.file-preview {
  margin-top: 20px;
}

.preview-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin: 0 0 16px 0;
}

.file-card {
  background-color: #F9FAFC;
  border: 1px solid #E4E7ED;
  border-radius: 6px;
  padding: 16px;
  margin-bottom: 12px;
  transition: all 0.3s ease;
}

.file-card:hover {
  border-color: #1677FF;
  box-shadow: 0 2px 4px rgba(22, 119, 255, 0.1);
}

.file-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.file-icon {
  color: #1677FF;
  font-size: 18px;
}

.file-name {
  flex: 1;
  font-weight: 500;
  color: #303133;
  font-size: 14px;
}

.file-size {
  color: #909399;
  font-size: 12px;
}

.remove-button {
  opacity: 0.7;
}

.remove-button:hover {
  opacity: 1;
}
</style>