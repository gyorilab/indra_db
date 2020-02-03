<template>
  <div id="app">
    <TimeView></TimeView>
    <AmountView></AmountView>
  </div>
</template>

<script>
import {TimeView, AmountView} from 'indralab-vue'

export default {
  name: 'app',
  components: {
    TimeView,
    AmountView
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
