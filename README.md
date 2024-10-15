# Learning SYCL (with Intel® oneAPI Base Toolkit)

I using Fedora Linux for development and using the [Data Parallel C++](https://link.springer.com/book/10.1007/978-1-4842-9691-2) Book for learning.

## Installing dependencies

- [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
- [Intel Compute Runtime](https://github.com/intel/compute-runtime)
- [CMake](https://cmake.org/)
- [Ninja](https://ninja-build.org/)

```sh
sudo dnf install intel-basekit intel-comupte-runtime cmake ninja-build
```

## Setup environment

```sh
# Must be bash or a bash-like shell
source /opt/intel/oneapi/setvars.sh
```

> [!INFO]
> Setup your editor to use the clangd provided by the Intel® oneAPI Base Toolkit
>
> Located at: /opt/intel/oneapi/compiler/latest/bin/compiler/clangd

## Generate compilation database

```sh
# At project root
mkdir build
cd build
cmake -GNinja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
cmake --build .
```
