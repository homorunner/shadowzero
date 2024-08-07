# This is a simple script to build for training / benchmarking / ...

# work in main directory
set -e

pushd cpp

if [ -d "subprojects/libtorch" ]; then
echo "Adding subprojects/libtorch to cmake prefix path..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/subprojects/libtorch/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)/subprojects/libtorch/lib # for mac
export LIBRARY_PATH=$LIBRARY_PATH:$(pwd)/subprojects/libtorch/lib
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$(pwd)/subprojects/libtorch
fi

if command -v nvcc; then
USE_CUDA=true
else
USE_CUDA=false
fi

# Debug
# meson setup -Dcpp_args="-march=native -Wno-unknown-pragmas" -Duse_cuda=$USE_CUDA ../build --buildtype=debugoptimized

# Release
meson setup -Dcpp_args="-ffast-math -march=native -Wno-unknown-pragmas" -Db_lto=true -Duse_cuda=$USE_CUDA ../build --buildtype=release

meson compile -C ../build

popd

pushd go
go mod tidy
CGO_ENABLED=1 GOAMD64=v3 go build -o ../build/trainhelper ./cmd/trainhelper
popd
