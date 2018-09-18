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

$BUILD_PATH = "build/solution"
$CMAKE_PATH = (Resolve-Path ../../vendor/grpc/third_party/protobuf/cmake).Path

pushd $CMAKE_PATH

# configure
if ((Test-Path ${BUILD_PATH}) -eq $False) { mkdir -p ${BUILD_PATH} }
pushd ${BUILD_PATH}
$relpath = (Resolve-Path -Relative $CMAKE_PATH)
cmake $relpath
popd

# build
cmake --build ./build/solution --config Debug
cmake --build ./build/solution --config Release

popd

exit 0
