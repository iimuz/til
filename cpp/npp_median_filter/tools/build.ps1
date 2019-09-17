<#
.SYNOPSIS
バイナリを作成します。

.DESCRIPTION
バイナリを作成します。
cmake.ps1 によりバイナリが生成されていることが前提となります。

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
.\build.ps1
バイナリを生成します。
.EXAMPLE
.\build.ps1 -clean
バイナリを再ビルドします。

.NOTES
None.
#>

[CmdletBinding(
  SupportsShouldProcess=$true,
  ConfirmImpact="Medium"
)]
Param(
  [string]$platform = "Release",
  [switch]$clean
)

# parameters
$SCRIPT_PATH = (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PROJECT_DIR = (Join-Path $SCRIPT_PATH "..")
$CMAKE_BUILD_DIR = (Join-Path $PROJECT_DIR "build")

# set option.
$option = @()
if ($clean) { $option += "--clean-first" }

# build
pushd $CMAKE_BUILD_DIR
& "C:\Program Files\CMake\bin\cmake.exe" --build . --config $platform $option
popd

