<template>
  <div class="monitor">
    <time-view></time-view>
    <amount-view></amount-view>
  </div>
</template>

<script>
  import TimeView from 'indralab-vue'
  import AmountView from 'indralab-vue'

  export default {
    name: "Monitor",
    components: {
      TimeView,
      AmountView
    },
  }
</script>

<style scoped>

</style>