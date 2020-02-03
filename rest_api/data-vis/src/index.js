import * as datavisComps from './components'

const install = (Vue) => {
  Object.values(datavisComps).forEach(comp => {
    window.console.log('DataVis installing ' + comp.name);
    Vue.use(comp);
  });
};

if (typeof window != 'undefined' && window.Vue) {
  install(window.Vue)
}

export default install

export {default as HelloWorld} from './components/HelloWorld'
export {default as Monitor} from './components/Monitor'
export {default as TimeView} from './components/TimeView'
export {default as AmountView} from './components/AmountView'
