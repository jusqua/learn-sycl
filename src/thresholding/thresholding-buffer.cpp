#include <filesystem>

#include <opencv2/imgcodecs.hpp>
#include <sycl/sycl.hpp>
#include <opencv2/opencv.hpp>
#include <fmt/core.h>

namespace fs = std::filesystem;

class thresholding;

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

    outfile += fmt::format("binary-{}", infile.filename().c_str());

    auto img = cv::imread(infile, cv::IMREAD_GRAYSCALE);
    if (img.empty() || !img.isContinuous() || img.dims != 2) {
        fmt::println(stderr, "[infile] is not a valid image file or unsupported format");
        return 4;
    }

    try {
        auto q = sycl::queue{};

        fmt::println("Method:   Buffer Data Management");
        fmt::println("Device:   {}", q.get_device().get_info<sycl::info::device::name>());
        fmt::println("Platform: {}", q.get_device().get_platform().get_info<sycl::info::platform::name>());

        constexpr uint8_t T = 128;
        auto size = sycl::range<1>{ img.elemSize() * img.total() };
        auto buf = sycl::buffer<uint8_t, 1>{ img.data, size };

        q.submit([&](sycl::handler& cgf) {
            auto acc = sycl::accessor{ buf, cgf, sycl::read_write, sycl::no_init };

            cgf.parallel_for<thresholding>(size, [=](sycl::id<1> i) {
                acc[i] = acc[i] > T ? 255 : 0;
            });
        });

        q.wait_and_throw();
    } catch (const sycl::exception& e) {
        fmt::println(stderr, "Exception caught: {}", e.what());
    }

    if (!cv::imwrite(outfile, img)) {
        fmt::println(stderr, "[outpath] has not enough disk space or permissions to write the output image");
        return 5;
    }
}
