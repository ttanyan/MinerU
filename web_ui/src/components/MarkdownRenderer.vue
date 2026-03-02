<template>
  <div class="markdown-renderer">
    <div v-if="content" class="markdown-content" v-html="renderedContent"></div>
    <div v-else class="empty-state">
      <el-empty :description="$t('results.noResults')" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import type { PropType } from 'vue'

const props = defineProps({
  content: {
    type: String as PropType<string>,
    default: ''
  }
})

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  breaks: true
})

// 添加LaTeX支持
md.use(function (md) {
  // 行内公式 $$...$$
  md.inline.ruler.after('escape', 'math_inline', function (state, silent) {
    let start = state.pos
    if (state.src.charCodeAt(start) !== 0x24/* $ */) return false
    if (state.src.charCodeAt(start + 1) !== 0x24/* $ */) return false

    let pos = start + 2
    let found = false

    while (pos < state.posMax) {
      if (state.src.charCodeAt(pos) === 0x24/* $ */ &&
          state.src.charCodeAt(pos + 1) === 0x24/* $ */) {
        found = true
        break
      }
      pos++
    }

    if (!found) return false

    if (!silent) {
      const token = state.push('math_inline', 'span', 0)
      token.content = state.src.slice(start + 2, pos)
      token.markup = '$$'
    }

    state.pos = pos + 2
    return true
  })

  // 行内公式 $...$
  md.inline.ruler.after('escape', 'math_inline_single', function (state, silent) {
    let start = state.pos
    if (state.src.charCodeAt(start) !== 0x24/* $ */) return false
    if (state.src.charCodeAt(start + 1) === 0x24/* $ */) return false // 避免与$$冲突

    let pos = start + 1
    let found = false

    while (pos < state.posMax) {
      if (state.src.charCodeAt(pos) === 0x24/* $ */) {
        found = true
        break
      }
      pos++
    }

    if (!found) return false

    if (!silent) {
      const token = state.push('math_inline_single', 'span', 0)
      token.content = state.src.slice(start + 1, pos)
      token.markup = '$'
    }

    state.pos = pos + 1
    return true
  })
})

// 自定义渲染器
md.renderer.rules.math_inline = function(tokens, idx) {
  return `<span class="math-inline">$$${tokens[idx].content}$$</span>`
}

md.renderer.rules.math_inline_single = function(tokens, idx) {
  return `<span class="math-inline">$${tokens[idx].content}$</span>`
}

const renderedContent = computed(() => {
  if (!props.content) return ''
  return md.render(props.content)
})
</script>

<style scoped>
.markdown-renderer {
  height: 100%;
  overflow-y: auto;
}

.markdown-content {
  padding: 20px;
  background: white;
  border-radius: 4px;
  min-height: 400px;
}

.markdown-content :deep(h1) {
  font-size: 2em;
  margin: 0.67em 0;
  font-weight: bold;
}

.markdown-content :deep(h2) {
  font-size: 1.5em;
  margin: 0.83em 0;
  font-weight: bold;
}

.markdown-content :deep(h3) {
  font-size: 1.17em;
  margin: 1em 0;
  font-weight: bold;
}

.markdown-content :deep(p) {
  margin: 1em 0;
  line-height: 1.6;
}

.markdown-content :deep(code) {
  background-color: #f5f7fa;
  padding: 2px 4px;
  border-radius: 3px;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
}

.markdown-content :deep(pre) {
  background-color: #f5f7fa;
  padding: 16px;
  border-radius: 4px;
  overflow-x: auto;
}

.markdown-content :deep(pre code) {
  background: none;
  padding: 0;
}

.markdown-content :deep(blockquote) {
  margin: 1em 0;
  padding: 0 1em;
  border-left: 4px solid #dcdfe6;
  color: #606266;
}

.markdown-content :deep(ul), .markdown-content :deep(ol) {
  margin: 1em 0;
  padding-left: 2em;
}

.markdown-content :deep(li) {
  margin: 0.5em 0;
}

.markdown-content :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
}

.markdown-content :deep(th), .markdown-content :deep(td) {
  border: 1px solid #dcdfe6;
  padding: 8px 12px;
  text-align: left;
}

.markdown-content :deep(th) {
  background-color: #f5f7fa;
  font-weight: bold;
}

.markdown-content :deep(img) {
  max-width: 100%;
  height: auto;
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 400px;
}

.math-inline {
  font-family: 'Cambria Math', 'STIX Two Math', serif;
  font-size: 1.1em;
}
</style>