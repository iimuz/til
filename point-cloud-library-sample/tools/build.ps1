<#
.SYNOPSIS
プロジェクトの操作を行います。
.DESCRIPTION
cmake が利用できることが前提となります。
プロジェクトの初期化、コンフィグ、ビルドが可能です。

.PARAMETER init
初期化を行う場合に付与します。
第三者モジュールの取得及びビルドを実行します。
一度実行した後は削除しない限りにおいて、付与しても再度生成することはありません。
.PARAMETER useProxy
proxy 環境下で初期化作業を行う場合に、 proxy 情報を入力するように設定します。
入力例) proxy.hoge.co.jp:port_number
.PARAMETER configure
コンフィグの設定を行います。
Visual Studio のソリューションファイルを生成します。
.PARAMETER noBuild
デフォルトではビルドを行いますが、ビルドを実行しない場合に付与します。
.PARAMETER platform
生成するバージョンを指定します。
Release または Debug になります。
.PARAMETER clean
生成済みの状態を削除してから各種処理を実施します。

.INPUTS
None. This script does not correspond.
.OUTPUTS
System.Int32
If success, this script returns 0, otherwise -1.

.EXAMPLE
.\build.ps1 -init -configure
最初に一度実行することで第三者モジュールを含めビルドします。

.EXAMPLE
.\build.ps1 -configure -clean
設定した状態を削除して再生成します。

.NOTES
None.
#>

[CmdletBinding(
  SupportsShouldProcess=$true,
  ConfirmImpact="Medium"
)]
Param(
  [switch]$init,
  [switch]$useProxy,
  [switch]$configure,
  [switch]$noBuild,
  [switch]$deploy,
  [string]$platform = "Release",
  [switch]$clean
)

# build
function Build {
  Param(
    [string]$platform = "",
    [bool]$clean = $False
  )
  begin {}
  process {
    $BUILD_DIR = (Get-Build-Dir)
    $CMAKE = (Get-CMake-Path)

    $option = @()
    if ($clean) { $option += "--clean-first" }

    pushd $BUILD_DIR
    & $CMAKE --build . --config $platform $option
    popd
  }
  end {}
}

# make clean directory
function Clean-Dir {
  Param(
    [bool]$clean = $False,
    [string]$dir = ""
  )
  begin {}
  process {
    if ($clean -And (Test-Path $dir)) {
      Write-Verbose "Remove: $dir"
      Remove-Item -Force -Recurse $dir
    }
    if ((Test-Path $dir) -eq $False) {
      Write-Verbose "Create: $dir"
      mkdir -p $dir | Out-Null
    }
  }
  end {}
}

# configure
function Configure {
  Param(
    [bool]$clean = $False
  )
  begin {}
  process {
    $BUILD_DIR = (Get-Build-Dir)
    $PROJ_DIR = (Get-Proj-Dir)
    $DEPLOY_DIR = (Get-Deploy-Dir)

    $CMAKE = (Get-CMake-Path)
    $VCPKG_CMAKE = (Get-Vcpkg-CMake-Path)
    $GENERATOR = "Visual Studio 15 2017 Win64"

    Clean-Dir -clean $clean -dir $BUILD_DIR
    pushd $BUILD_DIR
    & $CMAKE `
      (Resolve-Path -Relative $PROJ_DIR) `
      -G $GENERATOR `
      -DCMAKE_TOOLCHAIN_FILE="$VCPKG_CMAKE" `
      -DCMAKE_INSTALL_PREFIX="$DEPLOY_DIR"
    popd
  }
  end {}
}

# ビルドディレクトリの取得
function Get-Build-Dir {
  Param()
  begin {}
  process {
    $PROJ_DIR = (Get-Proj-Dir)
    $BUILD_DIR = (Join-Path $PROJ_DIR build)
    return $BUILD_DIR
  }
  end {}
}

# CMake のバイナリパスを取得
function Get-CMake-Path {
  Param()
  begin {}
  process {
    return "C:/Program Files/CMake/bin/cmake.exe"
  }
  end {}
}

# インストールディレクトリの取得
function Get-Deploy-Dir {
  Param()
  begin {}
  process {
    $PROJ_DIR = (Get-Proj-Dir)
    $INSTALL_DIR = (Join-Path $PROJ_DIR deploy)
    return $INSTALL_DIR
  }
  end {}
}

# プロジェクトのルートディレクトリを取得
function Get-Proj-Dir {
  Param()
  begin {}
  process {
    $SCRIPT_DIR = (Split-Path -Parent ${function:Get-Proj-Dir}.File)
    $PROJ_DIR = (Resolve-Path (Join-Path $SCRIPT_DIR ..)).Path
    return $PROJ_DIR
  }
  end {}
}

# vcpkg のルートディレクトリを取得
function Get-Vcpkg-Root-Dir {
  Param()
  begin {}
  process {
    $PROJ_DIR = (Get-Proj-Dir)
    $VCPKG_DIR = (Join-Path $PROJ_DIR vendor/vcpkg)
    return $VCPKG_DIR
  }
  end {}
}

# vcpkg のバイナリパスを取得
function Get-Vcpkg-Path {
  Param()
  begin {}
  process {
    $VCPKG_ROOT_DIR = (Get-Vcpkg-Root-Dir)
    $VCPKG = (Join-Path $VCPKG_ROOT_DIR vcpkg.exe)
    return $VCPKG
  }
  end {}
}

# vcpkg の cmake ファイルのパスを取得
function Get-Vcpkg-CMake-Path {
  Param()
  begin {}
  process {
    $VCPKG_ROOT_DIR = (Get-Vcpkg-Root-Dir)
    $VCPKG_CMAKE = (Join-Path $VCPKG_ROOT_DIR scripts/buildsystems/vcpkg.cmake)
    return $VCPKG_CMAKE
  }
  end {}
}

# 実行に必要なファイル群をまとめる
function Deploy {
  Param(
    [string]$platform = "",
    [bool]$clean = $False
  )
  begin {}
  process {
    $BUILD_DIR = (Get-Build-Dir)
    $DEPLOY_DIR = (Get-Deploy-Dir)

    $CMAKE = (Get-CMake-Path)

    # CMake によるデプロイ
    Clean-Dir -clean $clean -dir $DEPLOY_DIR
    pushd $BUILD_DIR
    & $CMAKE --build . --target install --config $platform
    popd
  }
  end {}
}

# install dependent package using vcpkg
function Vcpkg-Install {
  Param(
    [string]$pkg = "",
    [bool]$useProxy = $False
  )
  begin {}
  process {
    $VCPKG = (Get-Vcpkg-Path)

    if ((Test-Path $vcpkg) -eq $False) {
      Write-Error "cannot file vcpkg: $vcpkg"
      return
    }

    if ($useProxy) {
      $PROXY_URL = (Read-Host "Input proxy url. e.g) hoge:8080")
      Start-Process -Verb RunAs -Wait powershell "netsh winhttp set proxy proxy-server=${PROXY_URL}"
      $env:HTTP_PROXY = "http://${PROXY_URL}/"
      $env:HTTPS_PROXY = "http://${PROXY_URL}/"
    }
    & $VCPKG install $pkg
    if ($useProxy) { Start-Process -Verb RunAs -Wait powershell 'netsh winhttp reset proxy' }
  }
  end {}
}

# process
if ($init) { Vcpkg-Install -pkg pcl:x64-windows -useProxy $useProxy }
if ($configure) { Configure -clean $clean }
if ($noBuild -eq $False) { Build -platform $platform -clean $clean }
if ($deploy) { Deploy -platform $platform -clean $clean }

exit 0
