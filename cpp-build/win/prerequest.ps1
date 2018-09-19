<#
.SYNOPSIS
grpc に必要な事前条件のソフトウェアをインストールします。

.DESCRIPTION
grpc に必要な事前条件のソフトウェアをインストールします。
インストールには管理者権限が必要となります。

.INPUTS
None. This script does not correspond.
.OUTPUTS
System.Int32
If success, this script returns 0, otherwise -1.

.EXAMPLE
.\prerequest.ps1
Install packages.

.NOTES
None.
#>

[CmdletBinding(
  SupportsShouldProcess=$true,
  ConfirmImpact="Medium"
)]
Param()

# install chocolatory
Start-Process -Verb RunAs powershell "Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"

# install packages
Start-Process -Verb RunAs powershell "choco install -y cmake"
Start-Process -Verb RunAs powershell "choco install -y activeperl"
Start-Process -Verb RunAs powershell "choco install -y golang"
Start-Process -Verb RunAs powershell "choco install -y yasm"

exit 0
