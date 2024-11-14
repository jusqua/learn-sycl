#ifndef VISIONSYCL_PROCESSING_HPP
#define VISIONSYCL_PROCESSING_HPP

#include <visionsycl/image.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {
namespace host {

void inversion(const Image& input, Image& output);
void grayscale(const Image& input, Image& output);
void threshold(const Image& input, Image& output, int threshold = 128, int top = 255);

}  // namespace host
namespace usm {

void inversion(sycl::queue q, const Image& input, Image& output);
void grayscale(sycl::queue q, const Image& input, Image& output);
void threshold(sycl::queue q, const Image& input, Image& output, int threshold = 128, int top = 255);

}  // namespace usm
namespace buffer {

void inversion(sycl::queue q, const Image& input, Image& output);
void grayscale(sycl::queue q, const Image& input, Image& output);
void threshold(sycl::queue q, const Image& input, Image& output, int threshold = 128, int top = 255);

}  // namespace buffer
}  // namespace visionsycl

#endif  // VISIONSYCL_PROCESSING_HPP