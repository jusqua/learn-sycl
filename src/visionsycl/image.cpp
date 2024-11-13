#include <visionsycl/image.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

// TODO: Get rid of stb_image loader and writer

namespace visionsycl {

Image::Image() {
    this->dimensions = 0;
    this->channels = 0;
    this->length = 0;
    this->step = nullptr;
    this->shape = nullptr;
    this->data = nullptr;
}

Image::Image(int width, int height, int channels) {
    constexpr auto dims = 2;
    this->channels = channels;
    this->dimensions = dims;

    this->step = new int[dims];
    this->step[0] = 0;
    this->step[1] = 0;

    this->shape = new int[dims];
    this->shape[0] = height;
    this->shape[1] = width;

    this->length = this->channels;
    for (size_t i = 0; i < this->dimensions; ++i)
        this->length *= this->shape[i];

    this->data = new unsigned char[this->length];
}

Image::~Image() {
    delete[] this->shape;
    delete[] this->step;
    delete[] this->data;
}

Image load_image(const char* filepath) {
    constexpr auto dims = 2;
    auto image = Image();

    int x, y, comp;
    auto data = stbi_load(filepath, &x, &y, &comp, NULL);

    image.channels = comp;
    image.dimensions = dims;

    image.step = new int[dims];
    image.step[0] = 0;
    image.step[1] = 0;

    image.shape = new int[dims];
    image.shape[0] = y;
    image.shape[1] = x;

    image.length = image.channels;
    for (size_t i = 0; i < image.dimensions; ++i)
        image.length *= image.shape[i];

    image.data = data;

    return image;
}

int save_image_as(const char* filepath, Image& image) {
    return stbi_write_jpg(filepath, image.shape[1], image.shape[0], image.channels, image.data, 100);
}

}  // namespace visionsycl