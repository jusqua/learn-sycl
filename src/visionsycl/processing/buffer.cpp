#include <visionsycl/processing.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {
namespace buffer {

void invertion(sycl::queue q, const Image& input, Image& output) {
    constexpr uint8_t mask = 255;
    auto inptr = sycl::malloc_device<uint8_t>(input.length, q);
    auto outptr = sycl::malloc_device<uint8_t>(output.length, q);

    auto inbuf = sycl::buffer<uint8_t, 1>{ input.data, input.length };
    auto outbuf = sycl::buffer<uint8_t, 1>{ output.data, output.length };

    q.submit([&](sycl::handler& cgf) {
        auto inacc = inbuf.get_access<sycl::access::mode::read>();
        auto outacc = inbuf.get_access<sycl::access::mode::write>();

        cgf.parallel_for(input.length, [=](sycl::id<1> i) {
            outacc[i] = mask - inacc[i];
        });
    });

    q.wait_and_throw();
}

}  // namespace buffer
}  // namespace visionsycl