#include <visionsycl/selector.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {

int usm_selector_v(const sycl::device& dev) {
    if (dev.has(sycl::aspect::usm_device_allocations))
        return 1;
    return -1;
}

int opencl_selector_v(const sycl::device& dev) {
    if (dev.get_backend() == sycl::backend::opencl) {
        if (dev.has(sycl::aspect::gpu))
            return 1;
        else
            return 0;
    }
    return -1;
}

int priority_backend_selector_v(const sycl::device& dev) {
    switch (dev.get_backend()) {
    case sycl::backend::ext_oneapi_cuda:
    case sycl::backend::ext_oneapi_hip:
        return 3;
    case sycl::backend::ext_oneapi_level_zero:
        return 2;
    case sycl::backend::opencl:
        return opencl_selector_v(dev);
    default:
        return -1;
    }
}

}  // namespace visionsycl
