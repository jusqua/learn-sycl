# Vision SYCL (with Intel® oneAPI)

Attempt to provide common image processing functions that runs on heterogeneous devices.

## Installing Dependencies

- C++ Development Tools
- [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [CMake](https://cmake.org/)
- [Ninja](https://ninja-build.org/)

### Windows

Download the dependencies from the websites listed [above](#installing-dependencies).

First install the `Visual Studio Build Tools`:
```sh
winget install Microsoft.VisualStudio.2022.BuildTools
```
Open `Visual Studio Installer`, select current installed version, click `Modify` button, select `Development with C++` checkbox and hit `Modify` button, and wait it finish.

For `oneAPI Base Toolkit` you need to install `Visual Studio Build Tools` before install `oneAPI Base Toolkit`.
Install the netinstaller from the website listed above, and follow the instructions.

For `CMake` and `Ninja`:
```sh
winget install Kitware.CMake Ninja-build.Ninja
```

### Linux

Installing `CMake`, `Ninja` and basic development package and tools via package manager:

#### Debian/Ubuntu
```sh
sudo apt install cmake ninja-build pkg-config build-essential
```

#### Fedora
```sh
sudo dnf install cmake ninja-build pkg-config
sudo dnf group install "Development Tools"
```

#### OpenSUSE
```sh
sudo zypper install cmake ninja pkg-config
sudo zypper install -t pattern devel_C_C++
```

Add the `oneAPI Base Toolkit` repo from the website listed [above](#installing-dependencies).
```sh
# Switch apt to dnf/zypper if aplicable
sudo apt install intel-oneapi-base-toolkit
```

## Installing SYCL Backends

### Windows

Backends for Intel hardware are installed by default, for NVIDIA and AMD see [bellow](#nvidia-and-amd).

### Linux

This is for Intel backend, for NVIDIA and AMD see [below](#nvidia-and-amd).

#### Debian/Ubuntu
```sh
sudo apt install intel-opencl libze1 libze-intel-gpu1
```

#### Fedora
```sh
sudo dnf install intel-compute-runtime oneapi-level-zero
```

#### OpenSUSE
```sh
sudo zypper install intel-opencl libze_intel_gpu1
```

### NVIDIA and AMD

See: (Codeplay)[https://developer.codeplay.com/] solutions.

## Setup environment

### Windows

Using `cmd`:
```sh
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --include-intel-llvm
```

### Linux

Using `bash`:
```sh
. /opt/intel/oneapi/setvars.sh --include-intel-llvm
```

## Building

After [set up your environment](#setup-environment), in this project root, run:
```sh
cmake --preset linux # or windows
cmake --build build
```
