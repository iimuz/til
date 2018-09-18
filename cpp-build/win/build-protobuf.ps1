<#
.SYNOPSIS
protobuf をビルドします。

.DESCRIPTION
cmake が利用できることが前提となります。

.INPUTS
None. This script does not correspond.
.OUTPUTS
System.Int32
If success, this script returns 0, otherwise -1.

.EXAMPLE
.\build-protobuf.ps1
Build Protobuf module.

.NOTES
None.
#>

[CmdletBinding(
  SupportsShouldProcess=$true,
  ConfirmImpact="Medium"
)]
Param()

cd ../../vendor/grpc/third_party/protobuf/cmake
mkdir -p ./build/solution
cmake ./build/solution
cmake --build ./build/solution --config Debug
cmake --build ./build/solution --config Release

exit 0
