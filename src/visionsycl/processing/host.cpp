#include <visionsycl/processing.hpp>

namespace visionsycl {
namespace host {

void invertion(const Image& input, Image& output) {
    for (size_t i = 0; i < input.length; ++i)
        output.data[i] = 255 - input.data[i];
}

}  // namespace host
}  // namespace visionsycl