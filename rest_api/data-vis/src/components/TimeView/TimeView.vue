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

    <div v-for="day_bundle in bars" :key="day_bundle.day">
      <hr>
      <div class="row">
        <div class="col-1">
          {{ day_bundle.day }}
        </div>
        <div class="col-11">
          <figure>
            <svg :height="day_bundle.bars.length * 5"
                 width="100%"
                 role="img"
                 class="chart">
              <g v-for="(bar, index) in day_bundle.bars"
                 :key="bar.key"
                 class="bar">
                <rect :x="bar.start + '%'"
                      :width="bar.width + '%'"
                      :y="index * 5"
                      height="4"
                      :fill="bar.color"></rect>
              </g>
            </svg>
          </figure>
        </div>
      </div>

    </div>
  </div>
</template>

<script>
  const color_pallett = {
      content: {
        pubmed: '#006600',
        pmc_oa: '#669900',
        manuscripts: '#666633',
        all: '#609060'
      },
      reading: {
        REACH: '#00cc99',
        SPARSER: '#003399',
        ISI: '#9999ff',
        TRIPS: '#0080ff',
        all: '#606090'
      },
      preassembly: {
        all: '#cc5050'
      },
    };

  export default {
    name: "TimeView",
    data: function() {
      return {
        lo: 0,
        date_data: [],
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

      days: function() {
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
        return this.date_data.slice(this.lo, this.hi);
      },

      bars: function() {
        const ret = [];
        let day_bundle;
        for (let day of this.days) {
          day_bundle = {day: day.day_str, bars: []};
          for (let [stage_name, stage_data] of Object.entries(day.times)) {
            for (let [flavor_name, times] of Object.entries(stage_data)) {
              if (Object.keys(stage_data).length > 1 && flavor_name === 'all')
                continue;

              for (let timespan of times) {
                day_bundle.bars.push({
                  key: stage_name + flavor_name + timespan[0] + '-' + timespan[1],
                  stage: stage_name,
                  flavor: flavor_name,
                  start: Math.max(0, timespan[0]/24 * 100),
                  width: Math.max(0, (timespan[1] - timespan[0])/24 * 100),
                  color: color_pallett[stage_name][flavor_name]
                })
              }
            }
          }
          ret.push(day_bundle);
        }
        return ret;
      }
    }
  }
</script>

<style scoped>
  button {
    margin: 2px;
  }
  .bar {
    height: 21px;
  }
</style>