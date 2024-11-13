#include <chrono>
#include <iostream>
#include <filesystem>

#include <queue.hpp>
#include <visionsycl/image.hpp>
#include <visionsycl/processing.hpp>
#include <visionsycl/selector.hpp>

namespace ch = std::chrono;
namespace fs = std::filesystem;
namespace vn = visionsycl;

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " [INPUT IMAGE] [OUTPUT PATH] [[ROUNDS] = 1000]" << std::endl;
        return 1;
    }

    auto rounds = 1000;
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

    {
        auto start = ch::high_resolution_clock::now();
        vn::host::inversion(input, output);
        auto end = ch::high_resolution_clock::now();
        auto delta = ch::duration_cast<ch::milliseconds>(end - start);
        std::cout << "host invertion took " << delta.count() << "ms" << std::endl;
    }
    {
        auto start = ch::high_resolution_clock::now();
        vn::usm::inversion(queue, input, output);
        auto end = ch::high_resolution_clock::now();
        auto delta = ch::duration_cast<ch::milliseconds>(end - start);
        std::cout << "usm invertion took " << delta.count() << "ms" << std::endl;
    }
    {
        auto start = ch::high_resolution_clock::now();
        vn::buffer::inversion(queue, input, output);
        auto end = ch::high_resolution_clock::now();
        auto delta = ch::duration_cast<ch::milliseconds>(end - start);
        std::cout << "buffer invertion took " << delta.count() << "ms" << std::endl;
    }

    return 0;
}