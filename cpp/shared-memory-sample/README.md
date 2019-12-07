# Inter Process Communication Samples

# cpp

## shared memory(shm project)

### librt

shared memory を利用するにあたり、boostを利用しています。
boostのinterprocessはヘッダオンリーですが、利用するにはlibrtをリンクする必要があります。
リンクしない場合、下記のようなエラーとなるようです。

```txt
undefined reference to `shm_open'
```

