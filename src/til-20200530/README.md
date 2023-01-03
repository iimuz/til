# Autohot key 用スクリプト

## alt space ime

左 ALT + Space で 半角入力、
右 ALT + Space で全角入力となるようにショートカットキーを割り当てます。
元は、下記リポジトリを使っています。
下記リポジトリの場合、単一の左または右 ALT です。

- GitHub [karakaram/alt-ime-ahk][karakaram]
- その他の参考資料
  - 2016.2.13 [Windows10 64bit で Alt+Space による全角/半角切り替え][mocas_lab]
    - ALT + Space で切り替えられるが、トグルキーでしかないため、目的を満たさなかった。

[karakaram]: https://github.com/karakaram/alt-ime-ahk
[mocas_lab]: https://blog.goo.ne.jp/mocas_lab/e/3d1238365a243bb4614587076e159998

## loop ctrl shift

4分に1回 Ctrl, Shift を繰り返す処理を行います。

## loop mouse move

4分に1回マウス位置が上下左右に移動して、元の位置に戻る。