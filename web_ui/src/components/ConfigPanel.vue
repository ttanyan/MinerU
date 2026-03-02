<template>
  <el-card class="config-panel">
    <template #header>
      <div class="card-header">
        <span>{{ $t('config.title') }}</span>
      </div>
    </template>
    
    <el-form :model="config" label-position="top" label-width="120px">
      <!-- 最大页数 -->
      <el-form-item :label="$t('config.maxPages')">
        <el-slider 
          v-model="config.maxPages" 
          :min="1" 
          :max="1000" 
          show-input
        />
      </el-form-item>
      
      <!-- 解析后端 -->
      <el-form-item :label="$t('config.backend')">
        <el-select 
          v-model="config.backend" 
          style="width: 100%"
          @change="onBackendChange"
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
      >
        <el-input 
          v-model="config.serverUrl" 
          :placeholder="'http://localhost:30000'"
        />
        <div class="form-item-description">
          {{ $t('config.serverUrlInfo') }}
        </div>
      </el-form-item>
      
      <el-divider />
      
      <!-- 识别选项 -->
      <div class="section-title">{{ $t('config.recognitionOptions') }}</div>
      
      <!-- 表格识别 -->
      <el-form-item>
        <el-checkbox v-model="config.tableEnable">
          {{ $t('config.tableEnable') }}
        </el-checkbox>
        <div class="form-item-description">
          {{ $t('config.tableInfo') }}
        </div>
      </el-form-item>
      
      <!-- 公式识别 -->
      <el-form-item>
        <el-checkbox v-model="config.formulaEnable">
          {{ getFormulaLabel(config.backend) }}
        </el-checkbox>
        <div class="form-item-description">
          {{ getFormulaInfo(config.backend) }}
        </div>
      </el-form-item>
      
      <!-- OCR选项 (非VLM后端时显示) -->
      <template v-if="!isVlmBackend">
        <el-divider />
        
        <!-- OCR语言 -->
        <el-form-item :label="$t('config.ocrLanguage')">
          <el-select v-model="config.language" style="width: 100%">
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
        <el-form-item>
          <el-checkbox v-model="config.forceOcr">
            {{ $t('config.forceOcr') }}
          </el-checkbox>
          <div class="form-item-description">
            {{ $t('config.forceOcrInfo') }}
          </div>
        </el-form-item>
      </template>
    </el-form>
  </el-card>
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
  margin-bottom: 20px;
}

.card-header {
  font-weight: 500;
  font-size: 16px;
}

.section-title {
  font-weight: 500;
  margin: 20px 0 15px 0;
  color: #303133;
}

.form-item-description {
  font-size: 12px;
  color: #909399;
  margin-top: 5px;
  line-height: 1.4;
}
</style>