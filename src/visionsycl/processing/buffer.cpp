#include <visionsycl/processing.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {
namespace buffer {

void inversion(sycl::queue q, const Image& input, Image& output) {
    constexpr uint8_t mask = 255;
    auto channels = input.channels;
    auto inbuf = sycl::buffer<uint8_t, 1>{ input.data, input.length };
    auto outbuf = sycl::buffer<uint8_t, 1>{ output.data, output.length };

    q.submit([&](sycl::handler& cgf) {
        auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
        auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);

        cgf.parallel_for(input.length / channels, [mask, channels, inacc, outacc](sycl::id<1> idx) {
            auto i = idx[0] * channels;

            outacc[i] = mask - inacc[i];
            outacc[i + 1] = mask - inacc[i + 1];
            outacc[i + 2] = mask - inacc[i + 2];
        });
    });

    q.wait_and_throw();
}

void grayscale(sycl::queue q, const Image& input, Image& output) {
    auto channels = input.channels;
    auto inbuf = sycl::buffer<unsigned char, 1>{ input.data, input.length };
    auto outbuf = sycl::buffer<unsigned char, 1>{ output.data, output.length };

    q.submit([&](sycl::handler& cgf) {
        auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
        auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);

        cgf.parallel_for(input.length / channels, [channels, inacc, outacc](sycl::id<1> idx) {
            auto i = idx[0] * channels;

            auto mean = (inacc[i] + inacc[i + 1] + inacc[i + 2]) / 3;
            outacc[i] = mean;
            outacc[i + 1] = mean;
            outacc[i + 2] = mean;
        });
    });

    q.wait_and_throw();
}

void threshold(sycl::queue q, const Image& input, Image& output, int threshold, int top) {
    auto channels = input.channels;
    auto inbuf = sycl::buffer<unsigned char, 1>{ input.data, input.length };
    auto outbuf = sycl::buffer<unsigned char, 1>{ output.data, output.length };

    q.submit([&](sycl::handler& cgf) {
        auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
        auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);

        cgf.parallel_for(input.length / channels, [channels, threshold, top, inacc, outacc](sycl::id<1> idx) {
            auto i = idx[0] * channels;

            auto bin = (inacc[i] + inacc[i + 1] + inacc[i + 2]) / 3 > threshold ? top : 0;
            outacc[i] = bin;
            outacc[i + 1] = bin;
            outacc[i + 2] = bin;
        });
    });

    q.wait_and_throw();
}

}  // namespace buffer
}  // namespace visionsycl