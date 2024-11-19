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
    auto shape2d = sycl::range<2>{ static_cast<size_t>(input.shape[0]), static_cast<size_t>(input.shape[1]) };

    auto inptr = sycl::malloc_device<uint8_t>(input.length, queue);
    auto outptr = sycl::malloc_device<uint8_t>(output.length, queue);
    queue.memcpy(inptr, input.data, output.length).wait();

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
            for (size_t i = 0; i < input.length; i += input.channels) {
                output.data[i] = 255 - input.data[i];
                output.data[i + 1] = 255 - input.data[i + 1];
                output.data[i + 2] = 255 - input.data[i + 2];
            }
        };
        title = "host";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        constexpr unsigned char mask = 255;
        auto f = [&queue, &channels, &shape, &inptr, &outptr] {
            auto kernel = vn::InversionKernel<unsigned char*, unsigned char*, unsigned char>(channels, inptr, outptr, mask);
            queue.parallel_for(shape, kernel);
            queue.wait_and_throw();
        };
        title = "usm";
        perform_benchmark(title, rounds, f);
        usm_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        constexpr unsigned char mask = 255;
        auto f = [&queue, &channels, &shape, &inbuf, &outbuf] {
            queue.submit([&](sycl::handler& cgf) {
                auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
                auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);
                auto kernel = vn::InversionKernel<sycl::accessor<unsigned char, 1, sycl::access::mode::read>, sycl::accessor<unsigned char, 1, sycl::access::mode::write>, unsigned char>(channels, inacc, outacc, mask);

                cgf.parallel_for(shape, kernel);
            });

            queue.wait_and_throw();
        };
        title = "buffer";
        perform_benchmark(title, rounds, f);
        buffer_save_image(get_filepath(group, title, inpath, outpath));
    }
    std::cout << std::endl;

    group = "grayscale";
    std::cout << group << std::endl;
    {
        auto f = [&input, &output] {
            for (size_t i = 0; i < input.length; i += input.channels) {
                auto mean = (input.data[i] + input.data[i + 1] + input.data[i + 2]) / 3;
                output.data[i] = mean;
                output.data[i + 1] = mean;
                output.data[i + 2] = mean;
            }
        };
        title = "host";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto f = [&queue, &channels, &shape, &inptr, &outptr] {
            auto kernel = vn::GrayscaleKernel<unsigned char*, unsigned char*>(channels, inptr, outptr);
            queue.parallel_for(shape, kernel);
            queue.wait_and_throw();
        };
        title = "usm";
        perform_benchmark(title, rounds, f);
        usm_save_image(get_filepath(group, title, inpath, outpath));
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
        buffer_save_image(get_filepath(group, title, inpath, outpath));
    }
    std::cout << std::endl;

    group = "threshold";
    std::cout << group << std::endl;
    {
        auto threshold = 128;
        auto top = 255;
        auto f = [&input, &output, &threshold, &top] {
            for (size_t i = 0; i < input.length; i += input.channels) {
                output.data[i] = input.data[i] > threshold ? top : 0;
                output.data[i + 1] = input.data[i + 1] > threshold ? top : 0;
                output.data[i + 2] = input.data[i + 2] > threshold ? top : 0;
            }
        };
        title = "host";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto f = [&queue, &channels, &shape, &inptr, &outptr] {
            auto kernel = vn::ThresholdKernel<unsigned char*, unsigned char*>(channels, inptr, outptr);
            queue.parallel_for(shape, kernel);
            queue.wait_and_throw();
        };
        title = "usm";
        perform_benchmark(title, rounds, f);
        usm_save_image(get_filepath(group, title, inpath, outpath));
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
        buffer_save_image(get_filepath(group, title, inpath, outpath));
    }
    std::cout << std::endl;

    group = "erode";
    constexpr unsigned char erode_mask[] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };
    constexpr int erode_mask_length = 9;
    constexpr int erode_mask_width = 3;
    constexpr int erode_mask_height = 3;
    constexpr unsigned char erode_max = 255;
    std::cout << group << std::endl;
    {
        auto& mask = erode_mask;
        auto& max = erode_max;
        int midx = erode_mask_width / 2;
        int midy = erode_mask_height / 2;

        auto f = [&] {
            for (int row = 0; row < input.shape[0]; ++row) {
                for (int col = 0; col < input.shape[1]; ++col) {
                    int counter = 0;
                    unsigned char r = max, g = max, b = max;
                    float sum = r + g + b;

                    for (int i = -midx; i <= midx; ++i) {
                        for (int j = -midy; j <= midy; ++j, ++counter) {
                            auto x = col + i;
                            auto y = row + j;

                            if (x >= 0 && x < input.shape[1] && y >= 0 && y < input.shape[0]) {
                                auto pos = (y * input.shape[1] + x) * channels;
                                float new_sum = input.data[pos] + input.data[pos + 1] + input.data[pos + 2];
                                if (mask[counter] != 0 && sum > new_sum) {
                                    r = input.data[pos];
                                    g = input.data[pos + 1];
                                    b = input.data[pos + 2];
                                    sum = new_sum;
                                }
                            }
                        }
                    }

                    auto pos = (row * input.shape[1] + col) * channels;
                    output.data[pos] = r;
                    output.data[pos + 1] = g;
                    output.data[pos + 2] = b;
                }
            }
        };

        title = "host";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto maskptr = sycl::malloc_device<unsigned char>(erode_mask_length, queue);
        queue.memcpy(maskptr, erode_mask, erode_mask_length);
        auto& mask_width = erode_mask_width;
        auto& mask_height = erode_mask_height;
        auto& max = erode_max;

        auto f = [&] {
            auto kernel = vn::ErodeKernel<unsigned char*, unsigned char*, unsigned char*, unsigned char>(channels, inptr, outptr, maskptr, mask_width, mask_height, max);
            queue.parallel_for(shape2d, kernel);
            queue.wait_and_throw();
        };

        title = "usm";
        perform_benchmark(title, rounds, f);
        usm_save_image(get_filepath(group, title, inpath, outpath));
        sycl::free(maskptr, queue);
    }
    {
        auto mask = sycl::buffer<unsigned char>{ erode_mask, erode_mask_length };
        auto& mask_width = erode_mask_width;
        auto& mask_height = erode_mask_height;
        auto& max = erode_max;

        auto f = [&] {
            queue.submit([&](sycl::handler& cgf) {
                auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
                auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);
                auto maskacc = sycl::accessor(outbuf, cgf, sycl::read_only);
                auto kernel = vn::ErodeKernel<sycl::accessor<unsigned char, 1, sycl::access::mode::read>, sycl::accessor<unsigned char, 1, sycl::access::mode::write>, sycl::accessor<unsigned char, 1, sycl::access::mode::read>, unsigned char>(channels, inacc, outacc, maskacc, mask_width, mask_height, max);

                cgf.parallel_for(shape2d, kernel);
            });

            queue.wait_and_throw();
        };

        title = "buffer";
        perform_benchmark(title, rounds, f);
        buffer_save_image(get_filepath(group, title, inpath, outpath));
    }
    std::cout << std::endl;

    group = "dilate";
    constexpr unsigned char dilate_mask[] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };
    constexpr int dilate_mask_length = 9;
    constexpr int dilate_mask_width = 3;
    constexpr int dilate_mask_height = 3;
    constexpr unsigned char dilate_min = 0;
    std::cout << group << std::endl;
    {
        auto& mask = dilate_mask;
        auto& min = dilate_min;
        int midx = dilate_mask_width / 2;
        int midy = dilate_mask_height / 2;

        auto f = [&] {
            for (int row = 0; row < input.shape[0]; ++row) {
                for (int col = 0; col < input.shape[1]; ++col) {
                    int counter = 0;
                    unsigned char r = min, g = min, b = min;
                    float sum = r + g + b;

                    for (int i = -midx; i <= midx; ++i) {
                        for (int j = -midy; j <= midy; ++j, ++counter) {
                            auto x = col + i;
                            auto y = row + j;

                            if (x >= 0 && x < input.shape[1] && y >= 0 && y < input.shape[0]) {
                                auto pos = (y * input.shape[1] + x) * channels;
                                float new_sum = input.data[pos] + input.data[pos + 1] + input.data[pos + 2];
                                if (mask[counter] != 0 && sum < new_sum) {
                                    r = input.data[pos];
                                    g = input.data[pos + 1];
                                    b = input.data[pos + 2];
                                    sum = new_sum;
                                }
                            }
                        }
                    }

                    auto pos = (row * input.shape[1] + col) * channels;
                    output.data[pos] = r;
                    output.data[pos + 1] = g;
                    output.data[pos + 2] = b;
                }
            }
        };

        title = "host";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto maskptr = sycl::malloc_device<unsigned char>(dilate_mask_length, queue);
        queue.memcpy(maskptr, dilate_mask, dilate_mask_length);
        auto& mask_width = dilate_mask_width;
        auto& mask_height = dilate_mask_height;
        auto& min = dilate_min;

        auto f = [&] {
            auto kernel = vn::DilateKernel<unsigned char*, unsigned char*, unsigned char*, unsigned char>(channels, inptr, outptr, maskptr, mask_width, mask_height, min);
            queue.parallel_for(shape2d, kernel);
            queue.wait_and_throw();
        };

        title = "usm";
        perform_benchmark(title, rounds, f);
        usm_save_image(get_filepath(group, title, inpath, outpath));
        sycl::free(maskptr, queue);
    }
    {
        auto mask = sycl::buffer<unsigned char>{ dilate_mask, dilate_mask_length };
        auto& mask_width = dilate_mask_width;
        auto& mask_height = dilate_mask_height;
        auto& min = dilate_min;

        auto f = [&] {
            queue.submit([&](sycl::handler& cgf) {
                auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
                auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);
                auto maskacc = sycl::accessor(outbuf, cgf, sycl::read_only);
                auto kernel = vn::DilateKernel<sycl::accessor<unsigned char, 1, sycl::access::mode::read>, sycl::accessor<unsigned char, 1, sycl::access::mode::write>, sycl::accessor<unsigned char, 1, sycl::access::mode::read>, unsigned char>(channels, inacc, outacc, maskacc, mask_width, mask_height, min);

                cgf.parallel_for(shape2d, kernel);
            });

            queue.wait_and_throw();
        };

        title = "buffer";
        perform_benchmark(title, rounds, f);
        buffer_save_image(get_filepath(group, title, inpath, outpath));
    }
    std::cout << std::endl;

    sycl::free(inptr, queue);
    sycl::free(outptr, queue);

    return 0;
}