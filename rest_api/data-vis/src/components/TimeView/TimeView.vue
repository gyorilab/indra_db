<template>
  <div class="time-view">
    <div align="center">
      <button class="btn btn-outline-dark"
              :disabled="!canDelta(-1)"
              v-on:click="changeDay(-1)">
        Previous
      </button>
      <button class="btn btn-outline-dark"
              :disabled="!canDelta(1)"
              v-on:click="changeDay(1)">
        Next
      </button>
    </div>

    <div>
      <span v-for="(flavors, stage) in color_pallett" :key="stage">
        <b>{{ stage }}</b>:
        <span v-for="(color, flavor) in flavors" :key="flavor">
          <span v-if="Object.keys(flavors).length <= 1 || flavor !== 'all'">
              <span class="legend-dot"
                    :style="`background-color: ${color};`"></span>
              {{ flavor.toLowerCase() }}
          </span>
        </span>
        &nbsp;&nbsp;
      </span>
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
                      @mouseover="showTooltip($event, bar)"
                      @mouseleave="tooltip_on = false"
                      :fill="bar.color"></rect>
              </g>
            </svg>
          </figure>
        </div>
      </div>
    </div>
    <div class="tooltip"
         :style="`opacity:${tooltip_on ? 1 : 0};
                  z-index:${tooltip_on ? 10 : -10};
                  left: ${tooltip.position.left}px;
                  top: ${tooltip.position.top}px`">
      <div><b>{{ tooltip.flavor }}</b></div>
      <div><span class="stage-label">{{ tooltip.stage }}</span>  {{ tooltip.start }} - {{ tooltip.stop }}</div>
    </div>

  </div>
</template>

<script>
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
    data: function() {
      return {
        lo: 0,
        color_pallett: {
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
        },
        date_data: [],
        tooltip_on: false,
        tooltip: {
          stage: '',
          flavor: '',
          start: '',
          stop: '',
          position: {left: 0, right: 0}
        }
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

      canDelta: function(delta) {
        return ((delta > 0 && this.hi + delta <= this.date_data.length)
            || (delta < 0 && this.lo + delta >= 0))
      },

      changeDay: function(delta) {
        if (!this.date_data.length)
          return;

        if (this.canDelta(delta))
          this.lo += delta;
      },

      showTooltip: function(event, bar) {
        this.tooltip_on = true;
        this.tooltip.stage = bar.stage;
        this.tooltip.flavor = bar.flavor.toLowerCase();
        this.tooltip.start = bar.start_time;
        this.tooltip.stop = bar.stop_time;
        this.tooltip.position = {left: event.pageX, top: event.pageY}
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
                  color: this.color_pallett[stage_name][flavor_name],
                  start_time: getTime(timespan[0]),
                  stop_time: getTime(timespan[1])
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
    cursor: pointer;
  }
  .legend-dot {
    display: inline-block;
    cursor: pointer;
    position: relative;
    width:12px;
    height:12px;
    border-radius:12px;
  }
  .tooltip {
    position: absolute;
    border-radius: 5px;
    background-color: white;
    padding: 5px;
    border: 1px solid #e3e3e3;
    transition: 0.2s ease all;
  }
  .stage-label {
    color: grey;
    font-weight: bold;
  }
</style>