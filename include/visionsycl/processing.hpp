#ifndef VISIONSYCL_PROCESSING_HPP
#define VISIONSYCL_PROCESSING_HPP

#include <visionsycl/image.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {
namespace host {

void inversion(const Image& input, Image& output);

}  // namespace host
namespace usm {

void inversion(sycl::queue q, const Image& input, Image& output);

}  // namespace usm
namespace buffer {

void inversion(sycl::queue q, const Image& input, Image& output);

}  // namespace buffer
}  // namespace visionsycl

#endif  // VISIONSYCL_PROCESSING_HPP