#include <visionsycl/processing.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {
namespace buffer {

void inversion(sycl::queue q, const Image& input, Image& output) {
    constexpr uint8_t mask = 255;
    auto inbuf = sycl::buffer<uint8_t, 1>{ input.data, input.length };
    auto outbuf = sycl::buffer<uint8_t, 1>{ output.data, output.length };

    q.submit([&](sycl::handler& cgf) {
        auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
        auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);

        cgf.parallel_for(input.length / 3, [mask, inacc, outacc](sycl::id<1> idx) {
            auto i = idx[0] * 3;

            outacc[i] = mask - inacc[i];
            outacc[i + 1] = mask - inacc[i + 1];
            outacc[i + 2] = mask - inacc[i + 2];
        });
    });

    q.wait_and_throw();
}

void grayscale(sycl::queue q, const Image& input, Image& output) {
    auto inbuf = sycl::buffer<unsigned char, 1>{ input.data, input.length };
    auto outbuf = sycl::buffer<unsigned char, 1>{ output.data, output.length };

    q.submit([&](sycl::handler& cgf) {
        auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
        auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);

        cgf.parallel_for(input.length / 3, [inacc, outacc](sycl::id<1> idx) {
            auto i = idx[0] * 3;

            auto mean = (inacc[i] + inacc[i + 1] + inacc[i + 2]) / 3;
            outacc[i] = mean;
            outacc[i + 1] = mean;
            outacc[i + 2] = mean;
        });
    });

    q.wait_and_throw();
}

}  // namespace buffer
}  // namespace visionsycl