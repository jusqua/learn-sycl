#ifndef VISIONSYCL_SELECTOR_HPP
#define VISIONSYCL_SELECTOR_HPP

#include <sycl/sycl.hpp>

namespace visionsycl {

int usm_selector_v(const sycl::device& dev);
int opencl_selector_v(const sycl::device& dev);

}  // namespace visionsycl

#endif  // VISIONSYCL_SELECTOR_HPP
