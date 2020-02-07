Vue.component('stmt-search', {
  template: `
    <div class='stmt_search'>
      <div>
        <input v-model="agent" placeholder="Enter agent here">
        <button @click='lookupOptions'>Ground</button>
        <span v-show='searching'>Searching...</span>
      </div>
      <div v-if="options">
        Gilda Options:
        <div v-for="(option, option_idx) in options" :key="option_idx">
          <div>
            {{ option.term.source }} - {{ option.term.norm_text }}
          </div>
            {{ option.term.status }} for {{ option.term.entity_name }}
          <div>
          </div>
        </div>
      </div>
    </div>
  `,
  data: function() {
    return {
      agent: null,
      searching: false,
      options: null,
    }
  },
  methods: {
    lookupOptions: async function() {
      this.searching = true;
      const resp = await fetch(`${this.$ground_url}?agent=${this.agent}`, {method: 'GET'})
      this.options = await resp.json();
      this.searching = false;
    },
  },
});
