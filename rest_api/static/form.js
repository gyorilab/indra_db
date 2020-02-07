Vue.component('stmt-search', {
  template: `
    <div class='stmt_search'>
      <div v-if="!options">
        <input v-model="agent" placeholder="Enter agent here">
        <button @click='lookupOptions'>Ground</button>
        <span v-show='searching'>Searching...</span>
      </div>
      <div v-else>
        <select v-model='selected_option'>
          <option v-if="options.length > 1" value='' selected disabled hidden>Select grounding option...</option>
          <option v-for='(option, option_idx) in options'
                  :key='option_idx'
                  :value='option_idx'
                  :selected="options.length > 1">
            <b>{{ option.term.entry_name }}</b> (score: {{ option.score.toFixed(2) }}, {{ option.term.status }} from {{ option.term.source }})
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
