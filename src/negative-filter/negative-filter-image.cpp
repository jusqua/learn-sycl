#include <filesystem>

#include <sycl/sycl.hpp>
#include <opencv2/opencv.hpp>
#include <fmt/core.h>

namespace fs = std::filesystem;

class negative_filter;

int image_selector(const sycl::device& dev) {
    if (dev.has(sycl::aspect::image))
        return 1;
    return -1;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fmt::println(stderr, "Usage: {} [infile] [outpath]", argv[0]);
        return 1;
    }

    fs::path infile(argv[1]);
    if (!infile.has_filename()) {
        fmt::println(stderr, "[infile] must be a image file");
        return 2;
    }

    fs::path outfile(argv[2]);
    if (outfile.has_filename()) {
        fmt::println(stderr, "[outpath] must be a path to output image file");
        return 3;
    }

    outfile += fmt::format("negative-{}", infile.filename().c_str());

    auto inimg = cv::imread(infile);
    if (inimg.empty() || !inimg.isContinuous() || inimg.dims != 2) {
        fmt::println(stderr, "[infile] is not a valid image file or unsupported format");
        return 4;
    }

    auto outimg = cv::Mat(inimg.rows, inimg.cols, inimg.type());

    try {
        auto q = sycl::queue{ image_selector };

        fmt::println("Method:   Image Data Management");
        fmt::println("Device:   {}", q.get_device().get_info<sycl::info::device::name>());
        fmt::println("Platform: {}", q.get_device().get_platform().get_info<sycl::info::platform::name>());

        constexpr sycl::uint4 mask{ 255 };
        sycl::range<2> img_range(inimg.rows, inimg.cols);
        sycl::unsampled_image<2> input_image(reinterpret_cast<sycl::uchar4*>(inimg.data), sycl::image_format::r8g8b8a8_uint, img_range);
        sycl::unsampled_image<2> output_image(reinterpret_cast<sycl::uchar4*>(outimg.data), sycl::image_format::r8g8b8a8_uint, img_range);

        q.submit([&](sycl::handler& cgh) {
            auto in_acc = input_image.get_access<sycl::uint4, sycl::access_mode::read>(cgh);
            auto out_acc = output_image.get_access<sycl::uint4, sycl::access_mode::write>(cgh);

            cgh.parallel_for<negative_filter>(img_range, [=](sycl::item<2> idx) {
                auto coord = sycl::int2(idx[0], idx[1]);
                out_acc.write(coord, mask - in_acc.read(coord));
            });
        });

        q.wait_and_throw();
    } catch (const sycl::exception& e) {
        fmt::println(stderr, "Exception caught: {}", e.what());
    }

    if (!cv::imwrite(outfile, outimg)) {
        fmt::println(stderr, "[outpath] has not enough disk space or permissions to write the output image");
        return 5;
    }
}
