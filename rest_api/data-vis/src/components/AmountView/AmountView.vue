<template>
  <div class="amount-view">
    <div class="form">
      <div>
        <multiselect v-model="stage"
                     :searchable="true"
                     :options="stages"
                     @input="getStageData"
                     placeholder="Select a stage"></multiselect>
      </div>
      <div v-if="measures">
        <multiselect v-model="selected_measures"
                     :options="measures"
                     :multiple="true"
                     placeholder="Select measure"></multiselect>
      </div>
    </div>
    <div v-for="measure in selected_measures"
         :key="measure">
      <h3>{{ measure }}</h3>
      <apexchart type="line"
                 height=300
                 :options="chartOptions"
                 :series="series[measure]">
      </apexchart>
    </div>
  </div>
</template>

<script>
  import VueApexCharts from 'vue-apexcharts'
  import Multiselect from 'vue-multiselect'

  export default {
    name: "AmountView",
    components: {
      apexchart: VueApexCharts,
      multiselect: Multiselect,
    },
    data: function() {
      return {
        // Stage info
        stages: [],
        stage: null,

        // Measurement option and selection.
        measures: null,
        selected_measures: [],

        // The backend data.
        amount_data: null,
        dates: null,

        // Plot options.
        chartOptions: {
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
              text: 'Count'
            }
          }
        },
      }
    },
    methods: {
      getStageData: async function() {
        /**
         * Retrieve the data file for the currently selected stage.
         *
         * This function sets the `amount_data` and `measures` attributes,
         * and also resets the `selected_measures` attributes. This function
         * does nothing if `stage` is empty/null.
         */
        // If there is no stage specified, there is nothing to do.
        if (!this.stage)
          return;

        // Reset the selected measures first
        this.selected_measures = null;

        // Get the data from the backend service.
        const resp = await fetch(
          this.$amount_view_url + this.stage,
          {method: 'GET'}
          );

        // Process the data.
        this.amount_data = await resp.json();
        this.measures = Object.keys(this.amount_data);

        // Figure out what dates are included in this file.
        let date_set = new Set();
        Object.values(this.amount_data).forEach(measure_data => {
          Object.values(measure_data).forEach(source_data => {
            Object.values(source_data).forEach(pair => {
              date_set.add(pair[0]);
            })
          })
        });
        this.dates = Array.from(date_set);
      },
      getStageOptions: async function() {
        /**
         * Get the list of possible stages from the backend server.
         *
         * This function populates the `stages` attribute.
         */
        const resp = await fetch(
          this.$amount_view_url + 'liststages',
          {method: 'GET'}
          );
        this.stages = await resp.json()
      }
    },
    created: function() {
      this.getStageOptions();
    },
    computed: {
      series: function() {
        /**
         * Compute the data series options.
         *
         * The data generated is based on the `amount_data` and `dates`
         * attributes.
         *
         * @return {Object} containing the line-plot data for each measure.
         */

        // If there is no amount data, there are no series
        if (!this.amount_data)
          return {};

        // Build up the return Object.
        let ret = {};
        let these_dates, these_data;
        for (let [measure, measure_data] of Object.entries(this.amount_data)) {
          ret[measure] = [];
          for (let [name, measures] of Object.entries(measure_data)) {
            // Get the data that exists within the measurements.
            these_dates = new Set();
            these_data = [];
            measures.forEach(pair => {
              these_dates.add(pair[0]);
              these_data.push(pair);
            });

            // Fill in the rest with nulls.
            this.dates.forEach(date => {
              if (!these_dates.has(date))
                these_data.push([date, null])
            });
            ret[measure].push({name: name, data: these_data});
          }
        }
        return ret
      }
    }
  }
</script>

<!-- <style src="vue-multiselect/dist/vue-multiselect.min.css"></style> -->

<style scoped>
  @import "~vue-multiselect/dist/vue-multiselect.min.css";

  .form {
    width: 50em;
    margin: 0 auto;
  }
</style>
