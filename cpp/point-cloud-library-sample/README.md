# Point Cloud Library Sample

This repository has PCL(Point Cloud Library) samples.
Sample code copy from [PCL Tutorial][pcl-tutorial].
Build settings of samples require Visual Studio 2017.

PCL のサンプルコードを Windows 環境 (Visual Studio 2017) で動作させるための実験を行っているリポジトリです。
サンプルコードは、 [PCL Tutorial][pcl-tutorial] からコピーしています。
ただし、必要に応じて修正している場合があります。

[pcl-tutorial]: http://pointclouds.org/documentation/tutorials/

# Build

tools フォルダ下にあるスクリプトを powershell で動作させることでビルドします。
PCL の導入には [vcpkg][vcpkg] を利用しています。
そのため、最初の 1 回だけは、 vcpkg を利用して PCL をビルドする必要があります。

ビルド手順は下記のようになります。

```ps1
$ git clone --recursive https://github.com/iimuz/point-cloud-library-sample.git pcl-samples
$ cd pcl-samples/vendor/vcpkg
$ powershell -ex bypass -f ./scripts/bootstrap.ps1
$ cd ../..

$ powershell -ex bypass -f ./tools/build.ps1 -init -configure  # 初回のみ
# or
$ powershell -ex bypass -f ./tools/build/ps1 -configure  # PCL のビルドはスキップ
```

(注意)

PCL ビルドには非常に多くの HDD 容量(約 40GB) と時間(環境によりますが、 4 時間程度)がかかります。
中間生成物に容量が取られているため、ビルド後は削除することで、軽量化可能です。
中間生成物は `vendor/vcpkg/buildtrees` に入っているため、フォルダごと削除できます。

[vcpkg]: https://github.com/Microsoft/vcpkg

# Commit

For the commit message, follow the method of [angular.js][angularjs].

[angularjs]: https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#commits

## Commit Message Format

Each commit message consists of a header,
a body and a footer. The header has a special format that includes a type,
a scope and a subject:

```txt
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

The header is mandatory and the scope of the header is optional.

Any line of the commit message cannot be longer 100 characters!
This allows the message to be easier to read on GitHub
as well as in various git tools.

## Type

Must be one of the following:

* feat: A new feature
* fix: A bug fix
* docs: Documentation only changes
* style: Changes that do not affect the meaning of the code (white-space,
    formatting, missing semi-colons, etc)
* refactor: A code change that neither fixes a bug nor adds a feature
* perf: A code change that improves performance
* test: Adding missing or correcting existing tests
* chore: Changes to the build process or auxiliary tools
    and libraries such as documentation generation

