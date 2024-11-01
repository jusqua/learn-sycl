#include <aspects.hpp>
#include <backend_types.hpp>
#include <cstdint>
#include <filesystem>

#include <info/info_desc.hpp>
#include <sycl/sycl.hpp>
#include <opencv2/opencv.hpp>
#include <fmt/core.h>

namespace fs = std::filesystem;

class negative_filter;

int usm_selector(const sycl::device& dev) {
    if (dev.has(sycl::aspect::usm_device_allocations))
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

    auto img = cv::imread(infile);
    if (img.empty() || !img.isContinuous() || img.dims != 2) {
        fmt::println(stderr, "[infile] is not a valid image file or unsupported format");
        return 4;
    }

    try {
        auto q = sycl::queue{ usm_selector };

        fmt::println("Method:   Unified Shared Memory Data Management");
        fmt::println("Device:   {}", q.get_device().get_info<sycl::info::device::name>());
        fmt::println("Platform: {}", q.get_device().get_platform().get_info<sycl::info::platform::name>());

        constexpr uint8_t mask = 255;
        auto size = img.elemSize1() * img.total();
        auto ptr = sycl::malloc_device<uint8_t>(size, q);

        auto load_device_ev = q.memcpy(ptr, img.data, size);

        auto filter_ev = q.parallel_for<negative_filter>(sycl::range{ size }, load_device_ev, [=](sycl::id<1> idx) {
            auto i = idx[0];
            ptr[i] = 255 - ptr[i];
        });

        auto load_host_ev = q.memcpy(img.data, ptr, size, filter_ev);

        load_host_ev.wait();
        sycl::free(ptr, q);

        q.throw_asynchronous();
    } catch (const sycl::exception& e) {
        fmt::println(stderr, "Exception caught: {}", e.what());
    }

    if (!cv::imwrite(outfile, img)) {
        fmt::println(stderr, "[outpath] has not enough disk space or permissions to write the output image");
        return 5;
    }
}
