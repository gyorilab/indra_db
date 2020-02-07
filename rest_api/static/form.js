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
        <select v-model='selected_option_idx'>
          <option :value='-1' selected disabled hidden>Select grounding option...</option>
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
      selected_option_idx: -1, 
    }
  },
  methods: {
    lookupOptions: async function() {
      this.searching = true;
      const resp = await fetch(`${this.$ground_url}?agent=${this.agent_str}`, {method: 'GET'})
      this.options = await resp.json();
      this.searching = false;

      if (this.options.length == 1)
        this.selected_option_idx = 0;
    },
    resetOptions: function() {
      this.options = null;
      this.selected_option_idx = -1;
    }
  },
  computed: {
    options_empty: function() {
      if (!this.options)
        return false
      return this.options.length == 0
    }
  },
  watch: {
    selected_option_idx: function(selected_option_idx) {
      if (selected_option_idx < 0)
        this.$emit('input', null);
      else
        this.$emit('input', this.options[selected_option_idx]);
    }
  }
});

Vue.component('stmt-search', {
  template: `
    <div class='stmt_search'>
      <div v-for="(agent, agent_idx) in agents" :key='agent_idx'>
        <agent-select v-model='agent.grounding'></agent-select>
      </div>
      <button @click='addAgent'>Add Agent</button>
    </div>`,
  data: function() {
    return {
      agents: [],
    }
  },
  methods: {
    addAgent: function() {
      this.agents.push({grounding: null})
    }
  },
  created: function() {
    this.addAgent();
  }
});
