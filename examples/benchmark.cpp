#include <chrono>
#include <filesystem>
#include <info/info_desc.hpp>
#include <iostream>

#include <visionsycl/image.hpp>
#include <visionsycl/processing.hpp>
#include <visionsycl/selector.hpp>

namespace ch = std::chrono;
namespace fs = std::filesystem;
namespace vn = visionsycl;

double get_delta(std::function<void(void)> func) {
    auto start = ch::high_resolution_clock::now();
    func();
    auto end = ch::high_resolution_clock::now();
    return static_cast<ch::duration<double, std::milli>>(end - start).count();
}

void perform_benchmark(fs::path inpath, fs::path outpath, size_t rounds, vn::Image &image, std::string title, std::vector<std::pair<std::string, std::function<void(void)>>> func_list) {
    double delta;

    std::cout << title << std::endl;
    for (auto func : func_list) {
        auto &name = func.first;
        auto &f = func.second;

        delta = get_delta(f);
        std::cout << name << ": " << delta << "ms (once) | ";
        delta = get_delta([&rounds, &f] { for (size_t i = 0; i < rounds; ++i) f(); });
        std::cout << delta << "ms (" << rounds << " times)" << std::endl;

        auto filepath = outpath.generic_string() + title + "-" + name + "-" + inpath.filename().generic_string();
        vn::save_image_as(filepath.c_str(), image);
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0]
                  << " [INPUT IMAGE] [OUTPUT PATH] [[ROUNDS] = 1000]" << std::endl;
        return 1;
    }

    size_t rounds = 1000;
    if (argc == 4) {
        auto arg = std::string(argv[3]);

        try {
            std::size_t pos;
            rounds = std::stoi(arg, &pos);

            if (pos < arg.size()) {
                std::cerr << "Error: [ROUNDS] not a number" << std::endl;
            }
        } catch (std::invalid_argument const &ex) {
            std::cerr << "Error: [ROUNDS] is an invalid argument" << std::endl;
        } catch (std::out_of_range const &ex) {
            std::cerr << "Error: [ROUNDS] is out of range" << std::endl;
        }
    }

    fs::path inpath(argv[1]);
    if (!inpath.has_filename()) {
        std::cerr << "Error: [INPUT IMAGE] must be an image file, e.g. JPG and PNG" << std::endl;
        return 2;
    }

    fs::path outpath(argv[2]);
    if (outpath.has_filename()) {
        std::cerr << "Error: [OUTPUT PATH] must be a path to output image file" << std::endl;
        return 3;
    }

    auto input = vn::load_image(inpath.generic_string().c_str());
    auto output = vn::Image(input.shape[1], input.shape[0], input.channels);
    auto queue = sycl::queue{ vn::usm_selector_v };

    std::cout << "Device:        " << queue.get_device().get_info<sycl::info::device::name>() << std::endl
              << "Platform:      " << queue.get_device().get_platform().get_info<sycl::info::platform::name>() << std::endl
              << "Compute Units: " << queue.get_device().get_info<sycl::info::device::max_compute_units>() << std::endl
              << std::endl;

    perform_benchmark(
        inpath,
        outpath,
        rounds,
        output,
        "inversion",
        {
            { "host", [&input, &output] { vn::host::inversion(input, output); } },
            { "usm", [&queue, &input, &output] { vn::usm::inversion(queue, input, output); } },
            { "buffer", [&queue, &input, &output] { vn::buffer::inversion(queue, input, output); } },
        });

    perform_benchmark(
        inpath,
        outpath,
        rounds,
        output,
        "grayscale",
        {
            { "host", [&input, &output] { vn::host::grayscale(input, output); } },
            { "usm", [&queue, &input, &output] { vn::usm::grayscale(queue, input, output); } },
            { "buffer", [&queue, &input, &output] { vn::buffer::grayscale(queue, input, output); } },
        });

    perform_benchmark(
        inpath,
        outpath,
        rounds,
        output,
        "threshold",
        {
            { "host", [&input, &output] { vn::host::threshold(input, output); } },
            { "usm", [&queue, &input, &output] { vn::usm::threshold(queue, input, output); } },
            { "buffer", [&queue, &input, &output] { vn::buffer::threshold(queue, input, output); } },
        });

    return 0;
}