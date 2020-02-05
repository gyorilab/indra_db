<template>
  <div class="line-chart">
    <h3>
      {{ name }}
      <button class='btn btn-outline-dark'
              @click="changeScale">
        {{ otherScale(scale) }}
      </button>
    </h3>
    <apexchart type="line"
               height=300
               :options="chartOptions"
               :series="series">
    </apexchart>
  </div>
</template>

<script>
  import VueApexCharts from 'vue-apexcharts'

  export default {
    name: "LineChart",
    components: {
      apexchart: VueApexCharts,
    },
    props: [
      'name',
      'data'
    ],
    data: function() {
      return {
        scale: 'Linear',
      }
    },
    methods: {
      changeScale: function() {
        this.scale = this.otherScale(this.scale);
      },
      otherScale: function(scale) {
        return (scale === 'Linear') ? 'Log' : 'Linear';
      },
      formatYLabel: function(value) {
        // If the value has been log-scaled, it will be a decimal.
        if (this.scale === 'Log')
          return Math.pow(10, value).toFixed(0);
        return value.toFixed(0);
      },
    },
    computed: {
      series: function() {
        if (this.scale === 'Linear')
          return this.data;

        const series = [];
        let scaled_line;
        for (let line of this.data) {
          scaled_line = {name: line.name, data: []};
          line.data.forEach(pair => {
            if (pair[1] > 0)
              scaled_line.data.push([pair[0], Math.log10(pair[1])]);
            else
              scaled_line.data.push([pair[0], null]);
          });
          series.push(scaled_line)
        }
        return series
      },
      chartOptions: function() {
        return {
          dataLabels: {
            enabled: false
          },
          stroke: {
            curve: 'straight',
          },
          grid: {
            padding: {
              right: 30,
              left: 20
            }
          },
          xaxis: {
            type: 'datetime',
            title: {
              text: 'Day',
            },
          },
          yaxis: {
            title: {
              text: (this.scale === 'Linear') ? 'Count' : 'Log Count'
            },
            labels: {
              formatter: this.formatYLabel
            }
          }
        }
      }
    }
  }
</script>

<style scoped>

</style>