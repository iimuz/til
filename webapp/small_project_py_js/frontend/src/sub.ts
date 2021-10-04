export const hello = (): void => {
  get_message()
};

function get_message(): void {
  document.getElementById("button").addEventListener("click", loadMessage);
}

function loadMessage(): void {
  var xhr = new XMLHttpRequest();
  xhr.open("GET", "/hello" + "?" + window.location.search.slice(1), true)
  xhr.onload = function() {
    if (this.status == 200) {
      var message = this.responseText;
      document.getElementById("comment").innerHTML = message;
    }
  };
  xhr.send();
}
