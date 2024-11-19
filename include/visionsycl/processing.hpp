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

template <typename inT, typename outT, typename maskT, typename T>
class ErodeKernel {
public:
    ErodeKernel(int channels, inT& in, outT& out, maskT& mask, int mask_width, int mask_height, int max) : channels(channels), in(in), out(out), mask(mask), midy(mask_height / 2), midx(mask_width / 2), max(max) {};
    void operator()(sycl::item<2> item) const {
        auto col = item.get_id(1);
        auto row = item.get_id(0);
        auto width = item.get_range(1);
        auto height = item.get_range(0);
        auto r = max, g = max, b = max;
        float sum = r + g + b;

        int counter = 0;
        for (int i = -midx; i <= midx; ++i) {
            for (int j = -midy; j <= midy; ++j, ++counter) {
                auto x = col + i;
                auto y = row + j;

                if (x >= 0 && x < width && y >= 0 && y < height) {
                    auto pos = (y * width + x) * channels;
                    float new_sum = in[pos] + in[pos + 1] + in[pos + 2];
                    if (mask[counter] != 0 && sum > new_sum) {
                        r = in[pos];
                        g = in[pos + 1];
                        b = in[pos + 2];
                        sum = new_sum;
                    }
                }
            }
        }

        auto pos = (row * width + col) * channels;
        out[pos] = r;
        out[pos + 1] = g;
        out[pos + 2] = b;
    }

private:
    int channels;
    inT in;
    outT out;
    maskT mask;
    int midx;
    int midy;
    T max;
};

template <typename inT, typename outT, typename maskT, typename T>
class DilateKernel {
public:
    DilateKernel(int channels, inT& in, outT& out, maskT& mask, int mask_width, int mask_height, int min) : channels(channels), in(in), out(out), mask(mask), midy(mask_height / 2), midx(mask_width / 2), min(min) {};
    void operator()(sycl::item<2> item) const {
        auto col = item.get_id(1);
        auto row = item.get_id(0);
        auto width = item.get_range(1);
        auto height = item.get_range(0);
        auto r = min, g = min, b = min;
        float sum = r + g + b;

        int counter = 0;
        for (int i = -midx; i <= midx; ++i) {
            for (int j = -midy; j <= midy; ++j, ++counter) {
                auto x = col + i;
                auto y = row + j;

                if (x >= 0 && x < width && y >= 0 && y < height) {
                    auto pos = (y * width + x) * channels;
                    float new_sum = in[pos] + in[pos + 1] + in[pos + 2];
                    if (mask[counter] != 0 && sum < new_sum) {
                        r = in[pos];
                        g = in[pos + 1];
                        b = in[pos + 2];
                        sum = new_sum;
                    }
                }
            }
        }

        auto pos = (row * width + col) * channels;
        out[pos] = r;
        out[pos + 1] = g;
        out[pos + 2] = b;
    }

private:
    int channels;
    inT in;
    outT out;
    maskT mask;
    int midx;
    int midy;
    T min;
};

}  // namespace visionsycl

#endif  // VISIONSYCL_PROCESSING_HPP