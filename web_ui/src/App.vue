<template>
  <div id="app">
    <el-container class="main-container">
      <el-main class="main-content">
        <DocumentProcessor />
      </el-main>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import DocumentProcessor from './views/DocumentProcessor.vue'
</script>

<style>
/* 全局样式 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  color: #303133;
  background-color: #F5F7FA;
}

#app {
  height: 100vh;
}
</style>

<style scoped>
.main-container {
  height: 100vh;
}

.main-content {
  padding: 40px;
  background-color: #F5F7FA;
}
</style>