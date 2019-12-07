export interface ISummarizeAPI {
  summarize(message: string): Promise<any>;
}

export class SummarizeA3rt implements ISummarizeAPI {
  private url: string = 'https://api.a3rt.recruit-tech.co.jp/text_summarization/v1/';
  private num: number = 2;
  private apikey: string = 'DZZARzegPMAIvcnAFIC6f2PAYyk0yjd5';

  public async summarize(message: string): Promise<string> {
    const formdata = new FormData();
    formdata.append('apikey', this.apikey);
    formdata.append('sentences', message);
    formdata.append('linenumber', String(this.num));

    const response
      = await fetch(this.url, {method: 'post', body: formdata})
        .catch((err: any) => err);
    const data = await response.json().catch((err: any) => err);
    return data.summary.join('。') + '。';
  }
}

