#ifndef VISIONSYCL_PROCESSING_HPP
#define VISIONSYCL_PROCESSING_HPP

#include <accessor.hpp>
#include <visionsycl/image.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {

template <typename inT, typename outT>
class InversionKernel {
public:
    InversionKernel(int channels, inT& in, outT& out) : channels(channels), in(in), out(out) {};
    SYCL_EXTERNAL void operator()(sycl::id<1> idx) const {
        auto i = idx[0] * channels;

        out[i] = mask - in[i];
        out[i + 1] = mask - in[i + 1];
        out[i + 2] = mask - in[i + 2];
    }

private:
    constexpr static unsigned char mask = 255;
    int channels;
    inT in;
    outT out;
};

template <typename inT, typename outT>
class GrayscaleKernel {
public:
    GrayscaleKernel(int channels, inT& in, outT& out) : channels(channels), in(in), out(out) {};
    SYCL_EXTERNAL void operator()(sycl::id<1> idx) const {
        auto i = idx[0] * channels;

        auto mean = (in[i] + in[i + 1] + in[i + 2]) / 3;
        out[i] = mean;
        out[i + 1] = mean;
        out[i + 2] = mean;
    }

private:
    int channels;
    inT in;
    outT out;
};

template <typename inT, typename outT>
class ThresholdKernel {
public:
    ThresholdKernel(int channels, inT& in, outT& out, int threshold = 128, int top = 255) : channels(channels), in(in), out(out), threshold(threshold), top(top) {};
    SYCL_EXTERNAL void operator()(sycl::id<1> idx) const {
        auto i = idx[0] * channels;

        auto bin = (in[i] + in[i + 1] + in[i + 2]) / 3 > threshold ? top : 0;
        out[i] = bin;
        out[i + 1] = bin;
        out[i + 2] = bin;
    }

private:
    int channels;
    int threshold;
    int top;
    inT in;
    outT out;
};

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