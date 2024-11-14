#include <visionsycl/processing.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {
namespace usm {

void inversion(sycl::queue q, const Image& input, Image& output) {
    constexpr uint8_t mask = 255;
    auto inptr = sycl::malloc_device<uint8_t>(input.length, q);
    auto outptr = sycl::malloc_device<uint8_t>(output.length, q);

    auto load_device_ev = q.memcpy(inptr, input.data, input.length);

    auto kernel_ev = q.parallel_for(sycl::range{ input.length / 3 }, { load_device_ev }, [mask, inptr, outptr](sycl::id<1> idx) {
        auto i = idx[0] * 3;

        outptr[i] = mask - inptr[i];
        outptr[i + 1] = mask - inptr[i + 1];
        outptr[i + 2] = mask - inptr[i + 2];
    });

    auto load_host_ev = q.memcpy(output.data, outptr, output.length, kernel_ev);

    load_host_ev.wait();
    sycl::free(inptr, q);
    sycl::free(outptr, q);
    q.throw_asynchronous();
}

void grayscale(sycl::queue q, const Image& input, Image& output) {
    auto inptr = sycl::malloc_device<uint8_t>(input.length, q);
    auto outptr = sycl::malloc_device<uint8_t>(output.length, q);

    auto load_device_ev = q.memcpy(inptr, input.data, input.length);

    auto kernel_ev = q.parallel_for(sycl::range{ input.length / 3 }, { load_device_ev }, [inptr, outptr](sycl::id<1> idx) {
        auto i = idx[0] * 3;

        auto mean = (inptr[i] + inptr[i + 1] + inptr[i + 2]) / 3;
        outptr[i] = mean;
        outptr[i + 1] = mean;
        outptr[i + 2] = mean;
    });

    auto load_host_ev = q.memcpy(output.data, outptr, output.length, kernel_ev);

    load_host_ev.wait();
    sycl::free(inptr, q);
    sycl::free(outptr, q);

    q.throw_asynchronous();
}

}  // namespace usm
}  // namespace visionsycl