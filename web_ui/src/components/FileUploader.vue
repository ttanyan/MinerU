<template>
  <div class="file-uploader">
    <el-upload
      drag
      :auto-upload="false"
      :show-file-list="true"
      :on-change="handleFileChange"
      :before-upload="beforeUpload"
      multiple
      accept=".pdf,.png,.jpg,.jpeg"
    >
      <el-icon class="el-icon--upload"><upload-filled /></el-icon>
      <div class="el-upload__text">
        {{ $t('upload.placeholder') }}
      </div>
      <template #tip>
        <div class="el-upload__tip">
          {{ $t('upload.supportedTypes') }} • {{ $t('upload.maxSize') }}
        </div>
      </template>
    </el-upload>
    
    <!-- 已上传文件预览 -->
    <div v-if="files.length > 0" class="file-preview">
      <el-card class="file-card" v-for="(file, index) in files" :key="index">
        <div class="file-info">
          <el-icon><document /></el-icon>
          <span class="file-name">{{ file.name }}</span>
          <span class="file-size">({{ formatFileSize(file.size) }})</span>
          <el-button 
            type="danger" 
            icon="delete" 
            circle 
            size="small"
            @click="removeFile(index)"
          />
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { UploadFilled, Document } from '@element-plus/icons-vue'
import type { UploadFile } from 'element-plus'

interface Props {
  modelValue: File[]
}

interface Emits {
  (e: 'update:modelValue', value: File[]): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const files = ref<File[]>(props.modelValue)

// 监听外部值变化
watch(() => props.modelValue, (newVal) => {
  files.value = newVal
})

// 监听内部值变化并通知父组件
watch(files, (newVal) => {
  emit('update:modelValue', newVal)
}, { deep: true })

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
  margin-bottom: 20px;
}

.file-preview {
  margin-top: 20px;
}

.file-card {
  margin-bottom: 10px;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.file-name {
  flex: 1;
  font-weight: 500;
}

.file-size {
  color: #909399;
  font-size: 12px;
}
</style>