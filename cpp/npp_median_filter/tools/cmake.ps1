<#
.SYNOPSIS
cmake を用いてソリューションファイルを生成する。

.DESCRIPTION
cmake を用いてソリューションファイルを生成する。

.PARAMETER clean
cmake フォルダを削除してから cmake を実行する。

.INPUTS
None. This script does not correspond.
.OUTPUTS
System.Int32
If success, this script returns 0, otherwise -1.

.EXAMPLE
.\cmake.ps1
cmake によるソリューションファイルの生成。

.EXAMPLE
.\cmake.ps1 -clean
以前の cmake 結果を削除して、再度ソリューションファイルを生成。

.NOTES
none
#>

[CmdletBinding(
  SupportsShouldProcess=$true,
  ConfirmImpact="Medium"
)]
Param(
  [switch]$clean
)

# Parameters.
$SCRIPT_PATH = (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PROJECT_DIR = (Join-Path $SCRIPT_PATH "..")
$CMAKE_BUILD_DIR = (Join-Path $PROJECT_DIR "build")
$CMAKE_GENERATOR = "Visual Studio 15 2017 Win64"

# Create cmake build directory.
if ($clean -And (Test-Path $CMAKE_BUILD_DIR)) {
  Write-Verbose "Remove: $CMAKE_BUILD_DIR"
  Remove-Item -Force -Recurse $CMAKE_BUILD_DIR
}
if ((Test-Path $CMAKE_BUILD_DIR) -eq $False) {
  Write-Verbose "Create: $CMAKE_BUILD_DIR"
  mkdir -p $CMAKE_BUILD_DIR | Out-Null
}

# Execute cmake.
pushd $CMAKE_BUILD_DIR
& "C:\Program Files\CMake\bin\cmake.exe" `
  (Resolve-Path -Relative $PROJECT_DIR) `
  -G $CMAKE_GENERATOR
popd

