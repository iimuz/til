<template>
  <div class="hello">
    <h1>A3RT Text Summarization API Sample</h1>
    <input class="input" type="text" id="apikey" placeholder="API Key">
    <textarea class="textarea" id="text" placeholder="Input sentences."></textarea>
    <a class="button is-primary" id="summarize">Summarize</a>
    <div class="box" id="result" style="margin-top: 20px">Output</div>
    <div class="content">
      Response status: <div id="res-code"></div>
      Response message: <div id="res-msg"></div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class HelloWorld extends Vue {}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style lang="sass" scoped>
@import "../../node_modules/bulma/bulma.sass";
</style>
