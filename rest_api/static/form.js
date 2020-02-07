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


Vue.component('agent-select', {
  template: `
    <div class='agent-select'>
      <div v-if="!options || options_empty">
        <input v-model="agent_str" placeholder="Enter agent here">
        <button @click='lookupOptions'>Ground</button>
        <span v-show='searching'>Searching...</span>
        <span v-show='options_empty'>No groundings found...</span>
      </div>
      <div v-else-if="options.length == 1">
        <span class='frozen-box'>
          <grounding-option v-bind="options[0]"></grounding-option>
        </span>
        <button @click='resetOptions'>Cancel</button>
      </div>
      <div v-else>
        <select :value='selected_option_idx'
                @input="$emit('input', options[selected_option_idx])">
          <option value='' selected disabled hidden>Select grounding option...</option>
          <option v-for='(option, option_idx) in options'
                  :key='option_idx'
                  :value='option_idx'>
            <grounding-option v-bind='option'></grounding-option>
          </option>
        </select>
        <button @click='resetOptions'>Cancel</button>
      </div>
    </div>`,
  props: ['value'],
  data: function() {
    return {
      agent_str: '',
      searching: false,
      options: null,
      selected_option: null
    }
  },
  methods: {
    lookupOptions: async function() {
      this.searching = true;
      const resp = await fetch(`${this.$ground_url}?agent=${this.agent_str}`, {method: 'GET'})
      this.options = await resp.json();
      this.searching = false;

      if (this.options.length == 1)
        this.$emit('input', this.options[0])
    },
    resetOptions: function() {
      this.options = null;
      this.selected_option = null;
      this.$emit('input', null);
    }
  },
  computed: {
    options_empty: function() {
      if (!this.options)
        return false
      return this.options.length == 0
    }
  },
});

Vue.component('stmt-search', {
  template: `
    <div class='stmt_search'>
      <agent-select v-model='agent'></agent-select>
    </div>`,
  data: function() {
    return {
      agent: '',
    }
  },
});
