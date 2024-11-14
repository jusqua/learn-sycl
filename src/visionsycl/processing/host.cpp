#include <visionsycl/processing.hpp>

namespace visionsycl {
namespace host {

void inversion(const Image& input, Image& output) {
    for (size_t i = 0; i < input.length; ++i)
        output.data[i] = 255 - input.data[i];
}

void grayscale(const Image& input, Image& output) {
    for (size_t i = 0; i < input.length; i += 3) {
        auto mean = (input.data[i] + input.data[i + 1] + input.data[i + 2]) / 3;
        output.data[i] = mean;
        output.data[i + 1] = mean;
        output.data[i + 2] = mean;
    }
}

}  // namespace host
}  // namespace visionsycl