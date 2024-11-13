#include <visionsycl/processing.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {
namespace buffer {

void invertion(sycl::queue q, const Image& input, Image& output) {
    constexpr uint8_t mask = 255;
    auto inbuf = sycl::buffer<uint8_t, 1>{ input.data, input.length };
    auto outbuf = sycl::buffer<uint8_t, 1>{ output.data, output.length };

    q.submit([&](sycl::handler& cgf) {
        auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
        auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);

        cgf.parallel_for(input.length, [mask, inacc, outacc](sycl::id<1> i) {
            outacc[i] = mask - inacc[i];
        });
    });

    q.wait_and_throw();
}

}  // namespace buffer
}  // namespace visionsycl