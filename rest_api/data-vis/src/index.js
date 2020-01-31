import * as dispComponents from './components'

const install = (Vue) => {
  Object.values(dispComponents).forEach(dispComponent => {
    Vue.use(dispComponent);
  })
};

if (typeof window != 'undefined' && window.Vue) {
  install(window.Vue)
}

export default install

export {default as HelloWorld} from './components/HelloWorld'
