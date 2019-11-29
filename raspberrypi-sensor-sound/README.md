# RaspberryPi Sensor Sound

ラズパイでセンサ入力を元にして音を鳴らします。

## Usage

下記を実行することで、起動状態となります。

```sh
pip install -r requirements.txt
python -m touch_sound
```

デフォルト設定の場合には音源は、mp3ファイルで`_test/sound`フォルダに配置してください。
配置されたファイルを読み取り自動で音源リストを生成します。

## ラズパイの音声回りのコマンド

* 音声デバイス選択: `sudo raspi-config` から audio を操作する。
* テスト音声再生: `speaker-test -t sine -f 600`
* ボリューム調整: `alsmixer`
