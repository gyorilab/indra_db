Vue.component('stmt-search', {
  template: `
    <div class='stmt_search'>
      <input v-model="agent" placeholder="Enter agent here">
      <button @click='lookupOptions'>Ground</button>
      <span v-show='searching'>Searching...</span>
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
