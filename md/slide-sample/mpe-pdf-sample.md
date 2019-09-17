---
presentation:
  theme: "white.css"
  width: 800
  height: 600
  center: true
  slideNumber: true
puppeteer:
  landscape: true
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

<!-- slide -->

# Markdown Preview Enahcned で スライド風 PDF

<!-- slide -->

## 概要

VSCode の Markdown Preview Enhanced 機能を利用して slide 環境を構築します。
Markdown Preview Enahnced は Reveal.js を利用してスライド表示しています。
また、 PDF 化の環境を整えておくことで、 PDF 出力も可能です。

<!-- slide -->

## Tips

<!-- slide -->

### Theme

front matter の `theme` タグに下記を設定可能です。
デフォルトでは `white.css` になっています。

beige.css, black.css, blood.css, league.css,
moon.css, night.css, serif.css, simple.css,
sky.css, solarized.css, white.css, none.css,
white.css

<!-- slide -->

### 改ページ

`<!-- slide -->` を挿入することで改ページが可能となります。

<!-- slide -->

### 文字の配置

front matter で `center: true` とすると、縦方向が中央寄せになります。

<!-- slide -->

### 文字の左寄せ

css で記載すれば文字列の左寄せが可能です。
最終的には、下記のような style.less を書くことでパラグラフ内の文字列が左寄せになります。

```less
.markdown-preview.markdown-preview {
  .reveal .slides {
    p {
      text-align: left;
    }
  }
}
```

- [Stack Overflow: How can I get left-justified paragraphs in reveal.js?][left-justified]

[left-justified]: https://stackoverflow.com/questions/21019476/how-can-i-get-left-justified-paragraphs-in-reveal-js

<!-- slide -->

### 縦方向スライド

縦方向にスライドを配置する場合は、 `<!-- slide vertical=true -->` を記載します。

<!-- slide vertical=true -->

このようにスライド方向が変わります。

<!-- slide -->

## 参考資料

- [Markdown Preview Enhanced Presentation][mpe-presentation]

[mpe-presentation]: https://shd101wyy.github.io/markdown-preview-enhanced/#/presentation?id=presentation-writer
