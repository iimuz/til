<template>
  <div class="hello">
    <h1>A3RT Text Summarization API Sample</h1>
    <textarea
      class="textarea"
      v-model="inputValue"
      placeholder="Input sentences.">
    </textarea>
    <a class="button is-primary" v-on:click="summarize">Summarize</a>
    <div class="box" v-model="resultValue" style="margin-top: 20px">
      {{ resultValue }}
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { summarizeController } from '@/summarizeController';

@Component
export default class HelloWorld extends Vue {
  private inputValue: string = '';
  private resultValue: string = 'Output';

  private async summarize(): Promise<any> {
    this.resultValue = await summarizeController(this.inputValue);
  }
}
</script>

<style lang="sass" scoped>
@import "../../node_modules/bulma/bulma.sass";
</style>
