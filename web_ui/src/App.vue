<template>
  <div id="app">
    <el-container class="main-container">
      <el-header class="header">
        <h1>{{ $t('app.title') }}</h1>
      </el-header>
      <el-main class="main-content">
        <DocumentProcessor />
      </el-main>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import DocumentProcessor from './views/DocumentProcessor.vue'
</script>

<style scoped>
.main-container {
  height: 100vh;
}

.header {
  background-color: #409eff;
  color: white;
  text-align: center;
  line-height: 60px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.header h1 {
  margin: 0;
  font-size: 24px;
}

.main-content {
  padding: 20px;
  background-color: #f5f7fa;
}
</style>