import * as dispComponents from './components'

const install = (Vue) => {
  Object.values(dispComponents).forEach(comp => {
    Vue.use(comp);
  })
};

if (typeof window != 'undefined' && window.Vue) {
  install(window.Vue)
}

export default install

export {default as HelloWorld} from './components/HelloWorld'
