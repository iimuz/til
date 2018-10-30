<#
.SYNOPSIS
dumpbin を利用した操作を行います。

.DESCRIPTION
dumpbin のオプションを指定して実行します。

.INPUTS
None. This script does not correspond.
.OUTPUTS
System.Int32
If success, this script returns 0, otherwise -1.

.EXAMPLE
.\dumpbin.ps1 /exports hoge.dll
hoge.dll で公開されている関数一覧を取得できます。

.EXAMPLE
.\dumpbin.ps1 /dependents hoge.dll
hoge.dll が依存するライブラリの一覧を取得できます。

.NOTES
None.
#>

[CmdletBinding(
  SupportsShouldProcess=$true,
  ConfirmImpact="Medium"
)]
Param(
  [string]$option = "",
  [string]$target = ""
)

$DUMPBIN = "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Tools\MSVC\14.15.26726\bin\Hostx64\x64\dumpbin.exe"

# dumpbin.exe が存在するかチェック
if ((Test-Path $DUMPBIN) -eq $False) {
  Write-Error "cannot find dumpbin.exe."
  exit -1
}

& $DUMPBIN $option $target

