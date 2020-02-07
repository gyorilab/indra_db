Vue.component('grounding-option', {
  template: `
    <span class='grounding-option'>
      <b>{{ term.entry_name }}</b> (score: {{ score.toFixed(2) }}, {{ term.status }} from {{ term.source }})
    </span>`,
  props: [
    'term',
    'score',
  ]
});


Vue.component('stmt-search', {
  template: `
    <div class='stmt_search'>
      <div v-if="!options">
        <input v-model="agent" placeholder="Enter agent here">
        <button @click='lookupOptions'>Ground</button>
        <span v-show='searching'>Searching...</span>
      </div>
      <div v-else-if="options.length == 1">
        <span class='frozen-box'>
          <grounding-option v-bind="options[0]"></grounding-option>
        </span>
        <button @click='resetOptions'>Cancel</button>
      </div>
      <div v-else>
        <select v-model='selected_option'>
          <option value='' selected disabled hidden>Select grounding option...</option>
          <option v-for='(option, option_idx) in options'
                  :key='option_idx'
                  :value='option_idx'>
            <grounding-option v-bind='option'></grounding-option>
          </option>
        </select>
        <button @click='resetOptions'>Cancel</button>
      </div>
    </div>
  `,
  data: function() {
    return {
      agent: null,
      searching: false,
      options: null,
      selected_option: null
    }
  },
  methods: {
    lookupOptions: async function() {
      this.searching = true;
      const resp = await fetch(`${this.$ground_url}?agent=${this.agent}`, {method: 'GET'})
      this.options = await resp.json();
      this.searching = false;
    },
    resetOptions: function() {
      this.options = null;
      this.selected_option = null;
    }
  },
});
