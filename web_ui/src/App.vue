<template>
  <div id="app">
    <div class="container">
      <DocumentProcessor />
    </div>
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
  color: #4E5969;
  background-color: #F2F3F5;
}

#app {
  height: 100%;
}

.container {
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  padding: 0;
  margin: 0;
}
</style>