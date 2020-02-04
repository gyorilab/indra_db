<template>
  <div class="time-view">
    <div align="center">
      <button class="btn btn-outline-dark"
              v-on:click="changeDay(-1)">
        Previous
      </button>
      <button class="btn btn-outline-dark"
              v-on:click="changeDay(1)">
        Next
      </button>
    </div>
    <apexchart
        type="rangeBar"
        height=300
        :options="chartOptions"
        :series="series">
    </apexchart>
  </div>
</template>

<script>
  import VueApexCharts from 'vue-apexcharts'

  function getTime(hours) {
    let minutes = Math.round((hours % 1) * 60);
    let min_str = '';
    if (minutes < 10)
      min_str = '0' + minutes;
    else
      min_str = minutes.toString();
    return Math.floor(hours) + ':' + min_str;
  }

  export default {
    name: "TimeView",
    components: {
      apexchart: VueApexCharts
    },
    data: function() {
      return {
        lo: 0,
        date_data: [],
        chartOptions: {
          chart: {
            height: 450,
            type: 'rangeBar'
          },
          colors: [
            function({seriesIndex, w}) {
              const [stage, flavor] = w.config.series[seriesIndex].name.split('-');
              switch (stage) {
                case 'content':
                  switch (flavor) {
                    case 'pubmed':
                      return '#006600';
                    case 'pmc_oa':
                      return '#669900';
                    case 'manuscripts':
                      return '#666633';
                  }
                  break;
                case 'reading':
                  switch (flavor) {
                    case 'REACH':
                      return '#00cc99';
                    case 'SPARSER':
                      return '#003399';
                    case 'ISI':
                      return '#9999ff';
                    case 'TRIPS':
                      return '#0080ff';
                  }
                  break;
                case 'preassembly':
                  return '#cc0000';
              }
              return '#808080';
            },
          ],
          tooltip: {
            custom: function({ seriesIndex, dataPointIndex, w }) {
              let dp = w.config.series[seriesIndex].data[dataPointIndex];
              return '<div class="apexcharts-tooltip-rangebar">' +
                '<div>' +
                  '<span class="series-name">' +
                    w.config.series[seriesIndex].name +
                  '</span>' +
                '</div>' +
                '<div>' +
                  '<span class="category">' +
                    dp.x + ' ' +
                  '</span>' +
                  '<span class="value">' +
                    getTime(dp.y[0]) + '-' + getTime(dp.y[1]) +
                  '</span>' +
                '</div>' +
                '</div>';
            }
          },
          plotOptions: {
            bar: {
              horizontal: true,
              barHeight: '80%',
            }
          },
          xaxis: {
            type: 'datetime',
            min: 0,
            max: 24
          },
          legend: {
            position: 'top',
            horizontalAlign: 'left'
          }
        },
      }
    },
    methods: {
      getDates: async function() {
        /**
         * Get the runtime data from the backend.
         */
        const resp = await fetch(this.$time_view_url, {method: 'GET'});
        this.date_data = await resp.json();
        this.lo = this.date_data.length - 3;
      },

      changeDay: function(delta) {
        if (!this.date_data.length)
          return;

        if ((delta > 0 && this.hi + delta <= this.date_data.length)
            || (delta < 0 && this.lo + delta >= 0))
          this.lo += delta;
      },
    },
    created: function() {
      this.getDates();
    },
    computed: {
      hi: function() {
        if (!this.date_data.length)
          return this.lo;

        return Math.min(this.lo + 3, this.date_data.length);
      },

      series: function() {
        /**
         * Generate the series data.
         *
         * This function re-organizes and selects a range of data from the
         * runtime JSON. Only 3 days are shown at a time, based on the `day`
         * prop passed to this component from on high.
         *
         * @return {Array} of objects with data for each stage.
         */
        if (!this.date_data.length)
          return [];

        // Declare a variable for the loop.
        let final_stage_name;

        // Build up a dictionary keyed by stage names.
        let ret = {};
        for (let day_obj of this.date_data.slice(this.lo, this.hi) ) {
          for (let [stage_name, stage_data] of Object.entries(day_obj['times'])) {
            for (let [flavor_name, times] of Object.entries(stage_data)) {

              // Build the stage name depending on what "flavors" are available
              if (Object.keys(stage_data).length > 1)
                // If there are multiple flavors within the stage, skip "all"
                // and otherwise append the flavor name.
                if (flavor_name === 'all')
                  continue;
                else
                  final_stage_name = stage_name  + '-' + flavor_name;
              else
                // Otherwise just use the stage name ("all" should in this case
                // be the ONLY flavor).
                final_stage_name = stage_name;

              // Check to see if this key is new.
              if ( !(final_stage_name in ret) )
                ret[final_stage_name] = {name: final_stage_name, data: []};

              // Add all the times to the data for this stage.
              for (let timespan of times) {
                ret[final_stage_name].data.push({
                  x: day_obj['day_str'],
                  y: timespan
                })
              }
            }
          }
        }

        // Return an array of objects.
        return Object.values(ret);
      }
    }
  }
</script>

<style scoped>
  button {
    margin: 2px;
  }
</style>