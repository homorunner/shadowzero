# work in main directory
set -e

pushd cpp

mkdir -p subprojects
meson wrap install gtest

popd