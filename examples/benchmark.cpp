#include <chrono>
#include <filesystem>
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

std::string get_filepath(std::string& group, std::string& title, fs::path& inpath, fs::path& outpath) {
    return outpath.generic_string() + group + "-" + title + "-" + inpath.filename().generic_string();
}

void perform_benchmark(std::string& title, size_t& rounds, std::function<void(void)> func) {
    double delta;
    delta = get_delta(func);
    std::cout << title << ": " << delta << "ms (once) | ";
    delta = get_delta([&rounds, &func] { for (size_t i = 0; i < rounds; ++i) func(); });
    std::cout << delta << "ms (" << rounds << " times)" << std::endl;
}

int main(int argc, char** argv) {
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
        } catch (std::invalid_argument const& ex) {
            std::cerr << "Error: [ROUNDS] is an invalid argument" << std::endl;
        } catch (std::out_of_range const& ex) {
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

    std::string group, title;
    auto queue = sycl::queue{ vn::usm_selector_v };
    auto input = vn::load_image(inpath.generic_string().c_str());
    auto output = vn::Image(input.shape[1], input.shape[0], input.channels);
    auto channels = input.channels;
    auto shape = input.length / input.channels;

    auto inptr = sycl::malloc_device<uint8_t>(input.length, queue);
    auto outptr = sycl::malloc_device<uint8_t>(output.length, queue);
    auto inbuf = sycl::buffer<uint8_t, 1>{ input.data, input.length };
    auto outbuf = sycl::buffer<uint8_t, 1>{ output.data, output.length };

    auto host_save_image = [&output](std::string filepath) {
        vn::save_image_as(filepath.c_str(), output);
    };
    auto usm_save_image = [&output, &outptr, &queue](std::string filepath) {
        queue.memcpy(output.data, outptr, output.length).wait();
        vn::save_image_as(filepath.c_str(), output);
    };
    auto buffer_save_image = [&output, &outbuf, &queue](std::string filepath) {
        auto outacc = outbuf.get_host_access();
        for (unsigned long i = 0; i < output.length; ++i)
            output.data[i] = outacc[i];
        vn::save_image_as(filepath.c_str(), output);
    };

    std::cout << "Device:        " << queue.get_device().get_info<sycl::info::device::name>() << std::endl
              << "Platform:      " << queue.get_device().get_platform().get_info<sycl::info::platform::name>() << std::endl
              << "Compute Units: " << queue.get_device().get_info<sycl::info::device::max_compute_units>() << std::endl
              << std::endl;

    group = "inversion";
    std::cout << group << std::endl;
    {
        auto f = [&input, &output] {
            vn::host::inversion(input, output);
        };
        title = "host";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto f = [&queue, &channels, &shape, &inptr, &outptr] {
            queue.submit([&](sycl::handler& cgf) {
                auto kernel = vn::InversionKernel<unsigned char*, unsigned char*>(channels, inptr, outptr);

                cgf.parallel_for(shape, kernel);
            });
            queue.wait_and_throw();
        };
        title = "usm";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto f = [&queue, &channels, &shape, &inbuf, &outbuf] {
            queue.submit([&](sycl::handler& cgf) {
                auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
                auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);
                auto kernel = vn::InversionKernel<sycl::accessor<unsigned char, 1, sycl::access::mode::read>, sycl::accessor<unsigned char, 1, sycl::access::mode::write>>(channels, inacc, outacc);

                cgf.parallel_for(shape, kernel);
            });

            queue.wait_and_throw();
        };
        title = "buffer";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    std::cout << std::endl;

    group = "grayscale";
    std::cout << group << std::endl;
    {
        auto f = [&input, &output] {
            vn::host::grayscale(input, output);
        };
        title = "host";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto f = [&queue, &channels, &shape, &inptr, &outptr] {
            auto kernel = vn::GrayscaleKernel<unsigned char*, unsigned char*>(channels, inptr, outptr);
            queue.submit([&](sycl::handler& cgf) {
                auto kernel = vn::GrayscaleKernel<unsigned char*, unsigned char*>(channels, inptr, outptr);

                cgf.parallel_for(shape, kernel);
            });
            queue.wait_and_throw();
        };
        title = "usm";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto f = [&queue, &channels, &shape, &inbuf, &outbuf] {
            queue.submit([&](sycl::handler& cgf) {
                auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
                auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);
                auto kernel = vn::GrayscaleKernel<sycl::accessor<unsigned char, 1, sycl::access::mode::read>, sycl::accessor<unsigned char, 1, sycl::access::mode::write>>(channels, inacc, outacc);

                cgf.parallel_for(shape, kernel);
            });

            queue.wait_and_throw();
        };
        title = "buffer";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    std::cout << std::endl;

    group = "threshold";
    std::cout << group << std::endl;
    {
        auto f = [&input, &output] {
            vn::host::threshold(input, output);
        };
        title = "host";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto f = [&queue, &channels, &shape, &inptr, &outptr] {
            auto kernel = vn::ThresholdKernel<unsigned char*, unsigned char*>(channels, inptr, outptr);
            queue.submit([&](sycl::handler& cgf) {
                auto kernel = vn::ThresholdKernel<unsigned char*, unsigned char*>(channels, inptr, outptr);

                cgf.parallel_for(shape, kernel);
            });
            queue.wait_and_throw();
        };
        title = "usm";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto f = [&queue, &channels, &shape, &inbuf, &outbuf] {
            queue.submit([&](sycl::handler& cgf) {
                auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
                auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);
                auto kernel = vn::ThresholdKernel<sycl::accessor<unsigned char, 1, sycl::access::mode::read>, sycl::accessor<unsigned char, 1, sycl::access::mode::write>>(channels, inacc, outacc);

                cgf.parallel_for(shape, kernel);
            });

            queue.wait_and_throw();
        };
        title = "buffer";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    std::cout << std::endl;

    return 0;
}