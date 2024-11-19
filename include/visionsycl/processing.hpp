#ifndef VISIONSYCL_PROCESSING_HPP
#define VISIONSYCL_PROCESSING_HPP

#include <visionsycl/image.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {

template <typename inT, typename outT, typename T>
class InversionKernel {
public:
    InversionKernel(int channels, inT& in, outT& out, T mask) : channels(channels), in(in), out(out), mask(mask) {};
    void operator()(sycl::id<1> idx) const {
        auto i = idx[0] * channels;

        out[i] = mask - in[i];
        out[i + 1] = mask - in[i + 1];
        out[i + 2] = mask - in[i + 2];
    }

private:
    int channels;
    inT in;
    outT out;
    T mask;
};

template <typename inT, typename outT>
class GrayscaleKernel {
public:
    GrayscaleKernel(int channels, inT& in, outT& out) : channels(channels), in(in), out(out) {};
    void operator()(sycl::id<1> idx) const {
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
    void operator()(sycl::id<1> idx) const {
        auto i = idx[0] * channels;

        out[i] = in[i] > threshold ? top : 0;
        out[i + 1] = in[i + 1] > threshold ? top : 0;
        out[i + 2] = in[i + 2] > threshold ? top : 0;
    }

private:
    int channels;
    int threshold;
    int top;
    inT in;
    outT out;
};

}  // namespace visionsycl

#endif  // VISIONSYCL_PROCESSING_HPP