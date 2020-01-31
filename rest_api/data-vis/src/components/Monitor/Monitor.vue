<template>
  <div class="monitor">
    <TimeView></TimeView>
    <AmountView></AmountView>
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