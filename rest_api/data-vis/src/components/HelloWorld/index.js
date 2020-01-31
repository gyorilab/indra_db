import dispComponent from './HelloWorld'

export default Vue => {
    Vue.component(dispComponent.name, dispComponent);
}