#ifndef VISIONSYCL_PROCESSING_HPP
#define VISIONSYCL_PROCESSING_HPP

#include <visionsycl/image.hpp>
#include <sycl/sycl.hpp>

namespace visionsycl {

template <typename inT, typename outT>
class InversionKernel {
public:
    InversionKernel(int channels, inT& in, outT& out)
        : channels(channels), in(in), out(out) {};

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
    static constexpr uint8_t mask = 255;
};

template <typename inT, typename outT>
class GrayscaleKernel {
public:
    GrayscaleKernel(int channels, inT& in, outT& out)
        : channels(channels), in(in), out(out) {};

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

template <typename inT, typename outT, typename T>
class ThresholdKernel {
public:
    ThresholdKernel(int channels, inT& in, outT& out, T control, T top)
        : channels(channels), in(in), out(out), control(control), top(top) {};

    void operator()(sycl::id<1> idx) const {
        auto i = idx[0] * channels;

        out[i] = in[i] > control ? top : 0;
        out[i + 1] = in[i + 1] > control ? top : 0;
        out[i + 2] = in[i + 2] > control ? top : 0;
    }

private:
    int channels;
    int control;
    int top;
    inT in;
    outT out;
};

template <typename inT, typename outT, typename maskT, typename T>
class ErodeKernel {
public:
    ErodeKernel(int channels, inT& in, outT& out, maskT& mask, int mask_width, int mask_height, int max)
        : channels(channels), in(in), out(out), mask(mask), midy(mask_height / 2), midx(mask_width / 2), max(max) {};

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
    DilateKernel(int channels, inT& in, outT& out, maskT& mask, int mask_width, int mask_height, int min)
        : channels(channels), in(in), out(out), mask(mask), midy(mask_height / 2), midx(mask_width / 2), min(min) {};

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

template <typename inT, typename outT, typename maskT, typename MaskScalarT, typename OutScalarT>
class ConvolutionKernel {
public:
    ConvolutionKernel(int channels, inT& in, outT& out, maskT& mask, int mask_width, int mask_height)
        : channels(channels), in(in), out(out), mask(mask), midy(mask_height / 2), midx(mask_width / 2) {};

    void operator()(sycl::item<2> item) const {
        auto col = item.get_id(1);
        auto row = item.get_id(0);
        auto width = item.get_range(1);
        auto height = item.get_range(0);
        MaskScalarT r = 0, g = 0, b = 0;

        int counter = 0;
        for (int i = -midx; i <= midx; ++i) {
            for (int j = -midy; j <= midy; ++j, ++counter) {
                auto x = col + i;
                auto y = row + j;

                if (x >= 0 && x < width && y >= 0 && y < height) {
                    auto pos = (y * width + x) * channels;
                    r += in[pos] * mask[counter];
                    g += in[pos + 1] * mask[counter];
                    b += in[pos + 2] * mask[counter];
                }
            }
        }

        auto pos = (row * width + col) * channels;
        out[pos] = static_cast<OutScalarT>(r);
        out[pos + 1] = static_cast<OutScalarT>(g);
        out[pos + 2] = static_cast<OutScalarT>(b);
    }

private:
    int channels;
    inT in;
    outT out;
    maskT mask;
    int midx;
    int midy;
    MaskScalarT min;
};

template <typename inT, typename outT, typename OutScalarT>
class GaussianBlur3X3Kernel {
public:
    GaussianBlur3X3Kernel(int channels, inT& in, outT& out)
        : channels(channels), in(in), out(out) {};

    void operator()(sycl::item<2> item) const {
        auto col = item.get_id(1);
        auto row = item.get_id(0);
        auto width = item.get_range(1);
        auto height = item.get_range(0);
        float r = 0, g = 0, b = 0;
        // clang-format off
        constexpr const static float mask[] = {
            1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
            2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
            1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
        };
        // clang-format on

        int counter = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j, ++counter) {
                auto x = col + i;
                auto y = row + j;

                if (x >= 0 && x < width && y >= 0 && y < height) {
                    auto pos = (y * width + x) * channels;
                    r += in[pos] * mask[counter];
                    g += in[pos + 1] * mask[counter];
                    b += in[pos + 2] * mask[counter];
                }
            }
        }

        auto pos = (row * width + col) * channels;
        out[pos] = static_cast<OutScalarT>(r);
        out[pos + 1] = static_cast<OutScalarT>(g);
        out[pos + 2] = static_cast<OutScalarT>(b);
    }

private:
    int channels;
    inT in;
    outT out;
};

}  // namespace visionsycl

#endif  // VISIONSYCL_PROCESSING_HPP
