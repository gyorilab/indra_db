<template>
  <div class="time-view">
    <button v-on:click="changeDay(-1)">Previous</button>
    <button v-on:click="changeDay(1)">Next</button>
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
          plotOptions: {
            bar: {
              horizontal: true,
              barHeight: '80%',
            }
          },
          xaxis: {
            type: 'datetime',
            min: 0,
            max: 28
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
      },

      changeDay: function(delta) {
        if (!this.date_data.length)
          return;

        if ((delta > 0 && this.hi + delta < this.date_data.length)
            || (delta < 0 && this.lo + delta > 0))
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
</style>