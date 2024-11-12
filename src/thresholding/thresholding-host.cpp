#include <cstdint>
#include <filesystem>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <fmt/core.h>

namespace fs = std::filesystem;

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

    fmt::println("Method:   Host Serial Execution");
    fmt::println("Device:   Host");
    fmt::println("Platform: Host");

    constexpr uint8_t T = 128;
    auto size = img.elemSize() * img.total();
    for (size_t i = 0; i < size; ++i) {
        img.data[i] = img.data[i] > T ? 255 : 0;
    }

    if (!cv::imwrite(outfile, img)) {
        fmt::println(stderr, "[outpath] has not enough disk space or permissions to write the output image");
        return 5;
    }
}
