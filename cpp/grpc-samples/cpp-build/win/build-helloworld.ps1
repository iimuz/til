<#
.SYNOPSIS
HelloWorld をビルドします。

.DESCRIPTION
cmake が利用できることが前提となります。
GRPC_AS_SUBMODULE を ON にしています。
ON にしない場合は、インストールした gRPC の場所を示す cmakefile が必要となるようです。

.INPUTS
None. This script does not correspond.
.OUTPUTS
System.Int32
If success, this script returns 0, otherwise -1.

.EXAMPLE
.\build-helloworld.ps1
Build HelloWorld project.

.NOTES
None.
#>

[CmdletBinding(
  SupportsShouldProcess=$true,
  ConfirmImpact="Medium"
)]
Param()

$BUILD_PATH = "build"
$CMAKE_PROJ_PATH = (Resolve-Path ../../vendor/grpc/examples/cpp/helloworld).Path
$CMAKE_TOOL_PATH = (Resolve-Path "C:/Program Files/CMake/bin").Path

$env:path += ${CMAKE_TOOL_PATH} + ";"

pushd $CMAKE_PROJ_PATH

# configure
if ((Test-Path ${BUILD_PATH}) -eq $False) { mkdir -p ${BUILD_PATH} }
pushd ${BUILD_PATH}
$relpath = (Resolve-Path -Relative $CMAKE_PROJ_PATH)
cmake $relpath -DGRPC_AS_SUBMODULE=ON
popd

# build
cmake --build ${BUILD_PATH} --config Debug
cmake --build ${BUILD_PATH} --config Release

popd

exit 0
