# NPP Median Filter

Nvidia Performance Primitives (NPP) を利用したメディアンフィルタの実行時間計測です。
Visual Studio 2017, CUDA10 で動作確認しています。

- [Nvidia Performacne Primitives (NPP)][npp]
- [NVIDIA 2D Image And Signal Performance Primitives (NPP)][npp-docs]

[npp]: https://developer.nvidia.com/npp
[npp-docs]: https://docs.nvidia.com/cuda/npp/index.html

## 使い方

cmake を利用してビルド環境を構築します。

```ps1
$ cd npp_median_filter/tools
$ powershell -ex bypass -f ./cmake.ps1
$ powershell -ex bypass -f ./build.dp1
$ ../build/Release/npp_median_filter.exe
```

## 実行結果例

- GPU: Nvidia GTX 1080Ti

```txt
cuda malloc time: 1.278 [ms]
prepare median buffer size: 5.613 [ms]
median filter time [0/ 10]: 160.509 [ms]
median filter time [1/ 10]: 1.265 [ms]
median filter time [2/ 10]: 1.222 [ms]
median filter time [3/ 10]: 1.198 [ms]
median filter time [4/ 10]: 1.199 [ms]
median filter time [5/ 10]: 1.249 [ms]
median filter time [6/ 10]: 2.86 [ms]
median filter time [7/ 10]: 1.193 [ms]
median filter time [8/ 10]: 1.201 [ms]
median filter time [9/ 10]: 1.218 [ms]
delete buffer: 0.74 [ms]
```
