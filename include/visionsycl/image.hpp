#ifndef VISIONSYCL_IMAGE_HPP
#define VISIONSYCL_IMAGE_HPP

namespace visionsycl {

class Image {
public:
    int channels;
    int dimensions;
    int* shape;
    int* step;
    unsigned char* data;

    Image();
    ~Image();
};

Image* load_image(const char* filepath);
int save_image_as(const char* filepath, Image* image);

}  // namespace visionsycl

#endif  // VISIONSYCL_IMAGE_HPP