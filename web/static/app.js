ELEMENT.locale(ELEMENT.lang.ja)
var app = new Vue({
  el: '#app',
  data: {
    tasks: [],
    newTask: "",
    loading: false,
  },
  created: function() {
    this.loading = false;
  },
  methods: {
    addTask: function(task) {
    },
  },
})
