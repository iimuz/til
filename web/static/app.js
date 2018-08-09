ELEMENT.locale(ELEMENT.lang.ja)
var app = new Vue({
  el: '#app',
  data: {
    tasks: [],
    newTask: "",
    loading: false,
  },
  created: function() {
    this.loading = true;
    axios.get('/tasks')
      .then((response) => {
        console.log(response);
        this.loading = false;
      })
      .catch((error) => {
        console.log(error)
        this.loading = false;
      })
  },
  methods: {
    addTask: function(task) {
    },
  },
})
