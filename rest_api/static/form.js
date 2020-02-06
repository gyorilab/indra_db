Vue.component('stmt-search', {
  template: `
    <div class='stmt_search'>
      <input v-model="agent" @input="typing = true" placeholder="Enter agent here">
      <span v-show='searching'>Searching...</span>
    </div>
  `,
  data: function() {
    return {
      agent: null,
      searching: false,
      options: null
    }
  },
  methods: {
    lookupOptions: async function() {
      const resp = await fetch(`${this.$ground_url}/${this.agent}`, {method: 'GET'})
      return await resp.json();
    },
  },
  watch: {
    message: _.debounce(function() {
        this.searching = true;
        this.options = this.lookupOptions();
        this.searching = false;
    }, 1000),
  }
});
