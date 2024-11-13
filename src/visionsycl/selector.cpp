#include <visionsycl/selector.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {

int usm_selector_v(const sycl::device& dev) {
    if (dev.has(sycl::aspect::usm_device_allocations))
        return 1;
    return -1;
}

}  // namespace visionsycl