---
puppeteer:
  landscape: false
  format: "A4"
  timeout: 3000
  scale: 1
  displayHeaderFooter: true
  headerTemplate: >
    <div id="header-template" style="font-size:10px !important; padding-left:5%">
      <span class="title"></span>
    </div>
  footerTemplate: >
    <div id="footer-template" style="font-size:10px !important; display:table; padding-left:5%; width:90%;">
      <span style="display:table-cell; text-align:left;" class="date"></span>
      <span style="display:table-cell; text-align:right;">
      <span class="pageNumber"></span> / <span class="totalPages"></span>
      </span>
    </div>
---

# Markdown Preview Enahcned で PDF

## 概要

VSCode で Markdown preview enchaned 拡張を有効にし、
chrome puppeteer を利用して PDF 化するサンプルです。

## Tips

### 紙面の縦方向、横方向

`landscape` の設定で変更できます。

- ture: 横方向
- false: 縦方向

### 改ページ

任意の位置で改ページを行うには、 `<div style="page-break-before:always"></div>` を挿入すればよい。
ただし、通常のプレビューなどでわかりやすくするためには css を当てたほうがきれいである。
例えば、水平線で改ページとする場合は、 style.less に下記のように記載する。

```less
.markdown-preview.markdown-preview {
  hr {
    border: none;
    page-break-after: always;
  }
}
```

---

## 参考資料

- [Markdown Preview Enhanced Puppeteer][mde-puppeteer]
- [puppeteer api][puppeteer]

[mde-puppeteer]: https://github.com/shd101wyy/markdown-preview-enhanced/blob/master/docs/puppeteer.md
[puppeteer]: https://github.com/GoogleChrome/puppeteer/blob/master/docs/api.md
