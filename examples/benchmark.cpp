#include <chrono>
#include <filesystem>
#include <iostream>

#include <nd_range.hpp>
#include <range.hpp>
#include <visionsycl/image.hpp>
#include <visionsycl/processing.hpp>
#include <visionsycl/selector.hpp>

namespace ch = std::chrono;
namespace fs = std::filesystem;
namespace vn = visionsycl;

double get_delta(std::function<void(void)> func);
std::string get_filepath(std::string& group, std::string& title, fs::path& inpath, fs::path& outpath);
void perform_benchmark(std::string& title, size_t& rounds, std::function<void(void)> func);

int main(int argc, char** argv) {
    size_t rounds = 1000;

    // Ensure correct number of arguments
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " [INPUT IMAGE] [OUTPUT PATH] [[ROUNDS] = " << rounds << "]" << std::endl;
        return 1;
    }

    // Ensure rounds is a number
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

    // Ensure input and output are valid
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

    // Load image from provided path
    auto input = vn::load_image(inpath.generic_string().c_str());
    auto output = vn::Image(input.shape[1], input.shape[0], input.channels);
    auto channels = input.channels;

    // Device definitions
    auto queue = sycl::queue{ vn::priority_backend_selector_v };
    auto subgroup_sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
    auto subgroup = subgroup_sizes.back();
    auto usm_compatible = queue.get_device().has(sycl::aspect::usm_device_allocations);

    // Image shape definitions
    auto shape = input.length / input.channels;
    auto shape2d = sycl::range<2>{ static_cast<size_t>(input.shape[0]), static_cast<size_t>(input.shape[1]) };

    // Image ND shape definitions
    auto global_work_size = sycl::range<2>{ static_cast<size_t>(input.shape[0]), static_cast<size_t>(input.shape[1]) };
    auto local_group_size = sycl::range<2>{ subgroup, subgroup };
    auto nd_range = sycl::nd_range<2>{ global_work_size, local_group_size };

    // USM image memory allocation
    // TODO: Move to a separate function to avoid code crash
    auto inptr = sycl::malloc_device<uint8_t>(input.length, queue);
    auto outptr = sycl::malloc_device<uint8_t>(output.length, queue);
    queue.memcpy(inptr, input.data, output.length).wait();

    // Buffer image memory allocation
    auto inbuf = sycl::buffer<uint8_t, 1>{ input.data, input.length };
    auto outbuf = sycl::buffer<uint8_t, 1>{ output.data, output.length };

    // Image save functions for every memory model
    // TODO: Move away from main function
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

    // Display device information
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Platform: " << queue.get_device().get_platform().get_info<sycl::info::platform::name>() << std::endl;
    std::cout << "Compute Units: " << queue.get_device().get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Possible Subgroup Sizes: ";
    for (auto size : subgroup_sizes) std::cout << size << " ";
    std::cout << std::endl;
    std::cout << "Selected Subgroup Size: " << subgroup << std::endl;
    std::cout << "USM Memory Model Compatible: " << (usm_compatible ? "Yes" : "No, desabling benchmarks") << std::endl;
    std::cout << std::endl;

    // Benchmarks
    group = "inversion";
    std::cout << group << std::endl;
    constexpr unsigned char inversion_mask = 255;
    {
        auto& mask = inversion_mask;
        auto f = [&] {
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
    if (usm_compatible) {
        auto& mask = inversion_mask;
        auto f = [&] {
            auto kernel = vn::InversionKernel<decltype(inptr), decltype(outptr), decltype(mask)>(channels, inptr, outptr, mask);
            queue.parallel_for(shape, kernel);
            queue.wait_and_throw();
        };
        title = "usm";
        perform_benchmark(title, rounds, f);
        usm_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto& mask = inversion_mask;
        auto f = [&] {
            queue.submit([&](sycl::handler& cgf) {
                auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
                auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);
                auto kernel = vn::InversionKernel<decltype(inacc), decltype(outacc), decltype(mask)>(channels, inacc, outacc, mask);

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
        auto f = [&] {
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
    if (usm_compatible) {
        auto f = [&] {
            auto kernel = vn::GrayscaleKernel<decltype(inptr), decltype(outptr)>(channels, inptr, outptr);
            queue.parallel_for(shape, kernel);
            queue.wait_and_throw();
        };
        title = "usm";
        perform_benchmark(title, rounds, f);
        usm_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto f = [&] {
            queue.submit([&](sycl::handler& cgf) {
                auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
                auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);
                auto kernel = vn::GrayscaleKernel<decltype(inacc), decltype(outacc)>(channels, inacc, outacc);

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
    constexpr unsigned char threshold_control = 128;
    constexpr unsigned char threshold_top = 255;
    std::cout << group << std::endl;
    {
        auto& control = threshold_control;
        auto& top = threshold_top;
        auto f = [&] {
            for (size_t i = 0; i < input.length; i += input.channels) {
                output.data[i] = input.data[i] > control ? top : 0;
                output.data[i + 1] = input.data[i + 1] > control ? top : 0;
                output.data[i + 2] = input.data[i + 2] > control ? top : 0;
            }
        };
        title = "host";
        perform_benchmark(title, rounds, f);
        host_save_image(get_filepath(group, title, inpath, outpath));
    }
    if (usm_compatible) {
        auto& control = threshold_control;
        auto& top = threshold_top;
        auto f = [&] {
            auto kernel = vn::ThresholdKernel<decltype(inptr), decltype(outptr), decltype(control)>(channels, inptr, outptr, control, top);
            queue.parallel_for(shape, kernel);
            queue.wait_and_throw();
        };
        title = "usm";
        perform_benchmark(title, rounds, f);
        usm_save_image(get_filepath(group, title, inpath, outpath));
    }
    {
        auto& control = threshold_control;
        auto& top = threshold_top;
        auto f = [&] {
            queue.submit([&](sycl::handler& cgf) {
                auto inacc = sycl::accessor(inbuf, cgf, sycl::read_only);
                auto outacc = sycl::accessor(outbuf, cgf, sycl::write_only, sycl::no_init);
                auto kernel = vn::ThresholdKernel<decltype(inacc), decltype(outacc), decltype(control)>(channels, inacc, outacc, control, top);

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
    if (usm_compatible) {
        auto maskptr = sycl::malloc_device<unsigned char>(erode_mask_length, queue);
        queue.memcpy(maskptr, erode_mask, erode_mask_length);
        auto& mask_width = erode_mask_width;
        auto& mask_height = erode_mask_height;
        auto& max = erode_max;

        auto f = [&] {
            auto kernel = vn::ErodeKernel<decltype(inptr), decltype(outptr), decltype(maskptr), decltype(max)>(channels, inptr, outptr, maskptr, mask_width, mask_height, max);
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
                auto kernel = vn::ErodeKernel<decltype(inacc), decltype(outacc), decltype(maskacc), decltype(max)>(channels, inacc, outacc, maskacc, mask_width, mask_height, max);

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
    if (usm_compatible) {
        auto maskptr = sycl::malloc_device<unsigned char>(dilate_mask_length, queue);
        queue.memcpy(maskptr, dilate_mask, dilate_mask_length);
        auto& mask_width = dilate_mask_width;
        auto& mask_height = dilate_mask_height;
        auto& min = dilate_min;

        auto f = [&] {
            auto kernel = vn::DilateKernel<decltype(inptr), decltype(outptr), decltype(maskptr), decltype(min)>(channels, inptr, outptr, maskptr, mask_width, mask_height, min);
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
                auto kernel = vn::DilateKernel<decltype(inacc), decltype(outacc), decltype(maskacc), decltype(min)>(channels, inacc, outacc, maskacc, mask_width, mask_height, min);

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
