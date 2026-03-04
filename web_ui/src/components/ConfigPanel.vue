<template>
  <div class="config-panel">
    <h3 class="panel-title">{{ $t('config.title') }}</h3>
    
    <el-form :model="config" label-position="top" label-width="120px" class="config-form">
      <!-- 最大页数 -->
      <el-form-item :label="$t('config.maxPages')" class="form-item">
        <el-slider 
          v-model="config.maxPages" 
          :min="1" 
          :max="1000" 
          show-input
          class="slider"
        />
      </el-form-item>
      
      <!-- 解析后端 -->
      <el-form-item :label="$t('config.backend')" class="form-item">
        <el-select 
          v-model="config.backend" 
          style="width: 100%"
          @change="onBackendChange"
          class="select"
        >
          <el-option
            v-for="option in backendOptions"
            :key="option.value"
            :label="option.label"
            :value="option.value"
          />
        </el-select>
        <div class="form-item-description">
          {{ getBackendDescription(config.backend) }}
        </div>
      </el-form-item>
      
      <!-- 服务器URL (仅当使用http-client时显示) -->
      <el-form-item 
        v-if="showServerUrl" 
        :label="$t('config.serverUrl')"
        class="form-item"
      >
        <el-input 
          v-model="config.serverUrl" 
          :placeholder="'http://localhost:30000'"
          class="input"
        />
        <div class="form-item-description">
          {{ $t('config.serverUrlInfo') }}
        </div>
      </el-form-item>
      
      <div class="divider"></div>
      
      <!-- 识别选项 -->
      <div class="section-title">{{ $t('config.recognitionOptions') }}</div>
      
      <!-- 表格识别 -->
      <el-form-item class="form-item">
        <el-checkbox v-model="config.tableEnable" class="checkbox">
          {{ $t('config.tableEnable') }}
        </el-checkbox>
        <div class="form-item-description">
          {{ $t('config.tableInfo') }}
        </div>
      </el-form-item>
      
      <!-- 公式识别 -->
      <el-form-item class="form-item">
        <el-checkbox v-model="config.formulaEnable" class="checkbox">
          {{ getFormulaLabel(config.backend) }}
        </el-checkbox>
        <div class="form-item-description">
          {{ getFormulaInfo(config.backend) }}
        </div>
      </el-form-item>
      
      <!-- OCR选项 (非VLM后端时显示) -->
      <template v-if="!isVlmBackend">
        <div class="divider"></div>
        
        <!-- OCR语言 -->
        <el-form-item :label="$t('config.ocrLanguage')" class="form-item">
          <el-select v-model="config.language" style="width: 100%" class="select">
            <el-option
              v-for="option in languageOptions"
              :key="option.value"
              :label="option.label"
              :value="option.value"
            />
          </el-select>
          <div class="form-item-description">
            {{ $t('config.ocrLanguageInfo') }}
          </div>
        </el-form-item>
        
        <!-- 强制OCR -->
        <el-form-item class="form-item">
          <el-checkbox v-model="config.forceOcr" class="checkbox">
            {{ $t('config.forceOcr') }}
          </el-checkbox>
          <div class="form-item-description">
            {{ $t('config.forceOcrInfo') }}
          </div>
        </el-form-item>
      </template>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { DocumentConfig } from '@/composables/useDocumentProcessor'

interface Props {
  modelValue: DocumentConfig
  backendOptions: Array<{ value: string; label: string }>
  languageOptions: Array<{ value: string; label: string }>
}

interface Emits {
  (e: 'update:modelValue', value: DocumentConfig): void
  (e: 'backendChange', backend: string): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const config = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

const showServerUrl = computed(() => {
  return props.modelValue.backend.includes('http-client')
})

const isVlmBackend = computed(() => {
  return props.modelValue.backend.startsWith('vlm')
})

const onBackendChange = (backend: string) => {
  emit('backendChange', backend)
}

const getFormulaLabel = (backend: string) => {
  if (backend.startsWith('vlm')) {
    return '启用行间公式识别'
  } else if (backend === 'pipeline') {
    return '启用公式识别'
  } else if (backend.startsWith('hybrid')) {
    return '启用行内公式识别'
  }
  return '启用公式识别'
}

const getFormulaInfo = (backend: string) => {
  if (backend.startsWith('vlm')) {
    return '禁用后，行间公式将显示为图片。'
  } else if (backend === 'pipeline') {
    return '禁用后，行间公式将显示为图片，行内公式将不会被检测或解析。'
  } else if (backend.startsWith('hybrid')) {
    return '禁用后，行内公式将不会被检测或解析。'
  }
  return ''
}

const getBackendDescription = (backend: string) => {
  const descriptions: Record<string, string> = {
    'pipeline': '传统多模型管道解析，支持多语言，无幻觉。',
    'vlm-auto-engine': '多模态大模型高精度解析，仅支持中英文文档。',
    'hybrid-auto-engine': '高精度混合解析，支持多语言。',
    'vlm-http-client': '多模态大模型高精度解析，通过远程服务器处理。',
    'hybrid-http-client': '高精度混合解析，通过远程服务器处理。'
  }
  return descriptions[backend] || '选择文档解析的后端引擎。'
}
</script>

<style scoped>
.config-panel {
  margin-bottom: 24px;
}

.panel-title {
  font-weight: 600;
  font-size: 16px;
  color: #303133;
  margin: 0 0 20px 0;
}

.config-form {
  width: 100%;
}

.form-item {
  margin-bottom: 20px;
}

.form-item label {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
  margin-bottom: 8px;
}

.divider {
  height: 1px;
  background-color: #E4E7ED;
  margin: 24px 0;
}

.section-title {
  font-weight: 500;
  margin: 0 0 16px 0;
  color: #303133;
  font-size: 14px;
}

.form-item-description {
  font-size: 12px;
  color: #909399;
  margin-top: 8px;
  line-height: 1.4;
}

.slider :deep(.el-slider__runway) {
  background-color: #E4E7ED;
}

.slider :deep(.el-slider__bar) {
  background-color: #1677FF;
}

.slider :deep(.el-slider__button) {
  border-color: #1677FF;
}

.select :deep(.el-select__input) {
  color: #303133;
}

.select :deep(.el-select__wrapper) {
  border: 1px solid #E4E7ED;
  border-radius: 4px;
  box-shadow: none;
}

.select :deep(.el-select__wrapper.is-focus) {
  border-color: #1677FF;
  box-shadow: 0 0 0 2px rgba(22, 119, 255, 0.2);
}

.input :deep(.el-input__wrapper) {
  border: 1px solid #E4E7ED;
  border-radius: 4px;
  box-shadow: none;
}

.input :deep(.el-input__wrapper.is-focus) {
  border-color: #1677FF;
  box-shadow: 0 0 0 2px rgba(22, 119, 255, 0.2);
}

.checkbox :deep(.el-checkbox__input.is-checked .el-checkbox__inner) {
  background-color: #1677FF;
  border-color: #1677FF;
}

.checkbox :deep(.el-checkbox__label) {
  color: #303133;
}
</style>