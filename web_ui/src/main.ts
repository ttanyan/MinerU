import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import { createI18n } from 'vue-i18n'
import App from './App.vue'
import { zh, en } from './locales'

const i18n = createI18n({
  legacy: false,
  locale: navigator.language.includes('zh') ? 'zh' : 'en',
  fallbackLocale: 'en',
  messages: {
    zh,
    en
  }
})

const app = createApp(App)
app.use(ElementPlus)
app.use(i18n)
app.mount('#app')