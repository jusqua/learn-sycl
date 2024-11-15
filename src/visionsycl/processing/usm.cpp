#include <visionsycl/processing.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {
namespace usm {

void inversion(sycl::queue q, const Image& input, Image& output) {
    constexpr uint8_t mask = 255;
    auto channels = input.channels;
    auto inptr = sycl::malloc_device<uint8_t>(input.length, q);
    auto outptr = sycl::malloc_device<uint8_t>(output.length, q);

    q.memcpy(inptr, input.data, input.length).wait();

    auto ev = q.parallel_for(input.length / channels, [mask, channels, inptr, outptr](sycl::id<1> idx) {
        auto i = idx[0] * channels;

        outptr[i] = mask - inptr[i];
        outptr[i + 1] = mask - inptr[i + 1];
        outptr[i + 2] = mask - inptr[i + 2];
    });

    q.memcpy(output.data, outptr, output.length, ev).wait();

    sycl::free(inptr, q);
    sycl::free(outptr, q);
    q.throw_asynchronous();
}

void grayscale(sycl::queue q, const Image& input, Image& output) {
    auto channels = input.channels;
    auto inptr = sycl::malloc_device<uint8_t>(input.length, q);
    auto outptr = sycl::malloc_device<uint8_t>(output.length, q);

    q.memcpy(inptr, input.data, input.length).wait();

    auto ev = q.parallel_for(input.length / channels, [channels, inptr, outptr](sycl::id<1> idx) {
        auto i = idx[0] * channels;

        auto mean = (inptr[i] + inptr[i + 1] + inptr[i + 2]) / 3;
        outptr[i] = mean;
        outptr[i + 1] = mean;
        outptr[i + 2] = mean;
    });

    q.memcpy(output.data, outptr, output.length, ev).wait();

    sycl::free(inptr, q);
    sycl::free(outptr, q);

    q.throw_asynchronous();
}

void threshold(sycl::queue q, const Image& input, Image& output, int threshold, int top) {
    auto channels = input.channels;
    auto inptr = sycl::malloc_device<uint8_t>(input.length, q);
    auto outptr = sycl::malloc_device<uint8_t>(output.length, q);

    q.memcpy(inptr, input.data, input.length).wait();

    auto ev = q.parallel_for(input.length / channels, [channels, threshold, top, inptr, outptr](sycl::id<1> idx) {
        auto i = idx[0] * 3;

        auto bin = (inptr[i] + inptr[i + 1] + inptr[i + 2]) / 3 > threshold ? top : 0;
        outptr[i] = bin;
        outptr[i + 1] = bin;
        outptr[i + 2] = bin;
    });

    q.memcpy(output.data, outptr, output.length, ev).wait();

    sycl::free(inptr, q);
    sycl::free(outptr, q);

    q.throw_asynchronous();
}

}  // namespace usm
}  // namespace visionsycl