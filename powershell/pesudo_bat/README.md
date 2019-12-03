# 疑似バッチファイル化

powershell スクリプトを実行するためには、実行許可を設定する必要があります。
簡単に実行してもらうためには、いくつかの方法がありますが、バッチファイル経由が多いようです。

## コマンド実行

実行時に権限を付与する方法としては、下記のように実行すればよいようです。
おそらくコマンドから実行できる場合は、この方法が良いと思います。

```ps1
powershell -ex bypass -f hoge.ps1
```

## 別のバッチファイルを用意

2つのファイルを用意し、バッチファイルからpowershellスクリプトを起動します。
ダブルクリックなどで実行できますが、2ファイル必要なため管理や受け渡しには不便かと思います。
hoge.ps1をhoge.batから実行します。

- dir
  - hoge.ps1: 実行したいスクリプト
  - hoge.bat: hoge.ps1を実行するためだけに利用する

hoge.bat

```bat
powershell -ex bypass -f hoge.ps1
```

## バッチファイルから自分自身をpowershellとして起動

拡張子は.batですが、中のスクリプトはpowershellで記載します。
サンプルコードは、 `sample.bat` になります。
ただし、最初の一行におまじないのように、自分自身をpowershellで呼び出すというバッチスクリプトを記載します。

```bat
@powershell -NoProfile -ExecutionPolicy Unrestricted "$s=[scriptblock]::create((gc \"%~f0\"|?{$_.readcount -gt 1})-join\"`n\");&$s" %*&goto:eof

Write-Output "This script writes using powershell."
```

おそらく本件の問題は、powershellスクリプト実行時に引数を設定できないことです。
ただ、配布目的であれば十分かもしれません。

