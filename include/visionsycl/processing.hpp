#ifndef VISIONSYCL_PROCESSING_HPP
#define VISIONSYCL_PROCESSING_HPP

#include <visionsycl/image.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {
namespace host {

void invertion(const Image& input, Image& output);

}  // namespace host
namespace usm {

void invertion(sycl::queue q, const Image& input, Image& output);

}  // namespace usm
namespace buffer {

void invertion(sycl::queue q, const Image& input, Image& output);

}  // namespace buffer
}  // namespace visionsycl

#endif  // VISIONSYCL_PROCESSING_HPP