import { createApp } from 'vue'
import App from './App.vue'
import {TimeView, AmountView} from './index'

const app = createApp(App)
app.use(TimeView)
app.use(AmountView)
app.mount('#app')
