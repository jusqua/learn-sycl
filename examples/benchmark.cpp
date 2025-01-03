#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <tuple>

#include <visionsycl/image.hpp>
#include <visionsycl/processing.hpp>
#include <visionsycl/selector.hpp>

namespace ch = std::chrono;
namespace fs = std::filesystem;
namespace vn = visionsycl;

int main(int argc, char** argv) {
    constexpr const size_t default_rounds = 1000;
    size_t rounds = default_rounds;

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
                rounds = default_rounds;
            }
        } catch (std::invalid_argument const& ex) {
            std::cerr << "Error: [ROUNDS] is an invalid argument" << std::endl;
            rounds = default_rounds;
        } catch (std::out_of_range const& ex) {
            std::cerr << "Error: [ROUNDS] is out of range" << std::endl;
            rounds = default_rounds;
        }
    }

    // Ensure input and output are valid
    fs::path inpath(argv[1]);
    if (!inpath.has_filename()) {
        std::cerr << "Error: [INPUT IMAGE] must be an image file, e.g. JPG or PNG" << std::endl;
        return 2;
    }
    fs::path outpath(argv[2]);
    if (outpath.has_filename()) {
        std::cerr << "Error: [OUTPUT PATH] must be a path to output image file" << std::endl;
        return 3;
    }

    // Device definitions
    auto q = sycl::queue{ vn::priority_backend_selector_v };
    auto is_usm_compatible = q.get_device().has(sycl::aspect::usm_device_allocations);

    // Display device information
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl
              << "Platform: " << q.get_device().get_platform().get_info<sycl::info::platform::name>() << std::endl
              << "Compute Units: " << q.get_device().get_info<sycl::info::device::max_compute_units>() << std::endl
              << "Memory Model: " << (is_usm_compatible ? "Unified Shared Memory" : "Generic Buffer") << std::endl
              << std::endl;

    if (!is_usm_compatible) {
        std::cerr << "Benchmark with Generic Buffer Memory Model is not possible yet." << std::endl;
        return -1;
    }

    // Benchmark function definitions
    std::vector<std::tuple<std::string, std::string, bool, std::function<void(void)>>> functions;

    // Load image from provided path
    auto input = vn::load_image(inpath.generic_string().c_str());
    auto output = vn::Image(input.shape[1], input.shape[0], input.channels);
    auto channels = input.channels;

    // Display Image information
    std::cout << "Image Dimensions: " << input.shape[1] << 'x' << input.shape[0] << std::endl
              << "Image Channels: " << input.channels << std::endl
              << "Image Length: " << input.length << " bytes" << std::endl
              << std::endl;

    // Image shape definitions
    auto linear_shape = input.length / input.channels;
    auto bidimensional_shape = sycl::range<2>{ static_cast<size_t>(input.shape[0]), static_cast<size_t>(input.shape[1]) };

    // Allocate memory for input and output images
    auto in = sycl::malloc_device<uint8_t>(input.length, q);
    auto out = sycl::malloc_device<uint8_t>(output.length, q);
    q.memcpy(in, input.data, input.length).wait_and_throw();

    // Generic save image
    auto save_func = [&output, &out, &q](std::string filepath) {
        q.memcpy(output.data, out, output.length).wait_and_throw();
        vn::save_image_as(filepath.c_str(), output);
    };

    // Load image to device
    auto load_to_device = [&input, &in, &q] {
        q.memcpy(in, input.data, input.length).wait_and_throw();
    };
    functions.push_back({ "Load Image to Device", "load-to-device", false, load_to_device });

    // Load image to host
    auto load_to_host = [&output, &out, &q] {
        q.memcpy(output.data, out, output.length).wait_and_throw();
    };
    functions.push_back({ "Load Image to Host", "load-to-host", false, load_to_host });

    // Inversion kernel
    auto inversion_kernel = vn::InversionKernel<decltype(in), decltype(out)>(channels, in, out);
    auto inversion = [&in, &out, &q, &linear_shape, &inversion_kernel] {
        q.parallel_for(linear_shape, inversion_kernel).wait_and_throw();
    };
    functions.push_back({ "Image Inversion", "inversion", true, inversion });

    // Grayscaling kernel
    auto grayscale_kernel = vn::GrayscaleKernel<decltype(in), decltype(out)>(channels, in, out);
    auto grayscale = [&in, &out, &q, &linear_shape, &grayscale_kernel] {
        q.parallel_for(linear_shape, grayscale_kernel).wait_and_throw();
    };
    functions.push_back({ "Image Grayscaling", "grayscale", true, grayscale });

    // Threshold kernel for binary image
    constexpr unsigned char threshold_control = 128;
    constexpr unsigned char threshold_top = 255;
    auto threshold_kernel = vn::ThresholdKernel<decltype(in), decltype(out), decltype(threshold_control)>(channels, in, out, threshold_control, threshold_top);
    auto threshold = [&in, &out, &q, &linear_shape, &threshold_kernel] {
        q.parallel_for(linear_shape, threshold_kernel).wait_and_throw();
    };
    functions.push_back({ "Image Thresholding", "threshold", true, threshold });

    // Erode kernel for cross masking
    constexpr unsigned char erode_mask_array[] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };
    constexpr int erode_mask_length = 9;
    constexpr int erode_mask_width = 3;
    constexpr int erode_mask_height = 3;
    constexpr unsigned char erode_max = 255;
    auto erode_mask = sycl::malloc_device<uint8_t>(erode_mask_length, q);
    q.memcpy(erode_mask, erode_mask_array, erode_mask_length);
    auto erode_kernel = vn::ErodeKernel<decltype(in), decltype(out), decltype(erode_mask), decltype(erode_max)>(channels, in, out, erode_mask, erode_mask_width, erode_mask_height, erode_max);
    auto erode = [&in, &out, &q, &bidimensional_shape, &erode_kernel] {
        q.parallel_for(bidimensional_shape, erode_kernel).wait_and_throw();
    };
    functions.push_back({ "Image Eroding (Cross Mask)", "erode", true, erode });

    // Dilate kernel for cross masking
    constexpr unsigned char dilate_mask_array[] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };
    constexpr int dilate_mask_length = 9;
    constexpr int dilate_mask_width = 3;
    constexpr int dilate_mask_height = 3;
    constexpr unsigned char dilate_min = 0;
    auto dilate_mask = sycl::malloc_device<uint8_t>(dilate_mask_length, q);
    q.memcpy(dilate_mask, dilate_mask_array, dilate_mask_length);
    auto dilate_kernel = vn::DilateKernel<decltype(in), decltype(out), decltype(dilate_mask), decltype(dilate_min)>(channels, in, out, dilate_mask, dilate_mask_width, dilate_mask_height, dilate_min);
    auto dilate = [&in, &out, &q, &bidimensional_shape, &dilate_kernel] {
        q.parallel_for(bidimensional_shape, dilate_kernel).wait_and_throw();
    };
    functions.push_back({ "Image Dilating (Cross Mask)", "dilate", true, dilate });

    // clang-format off
    // Convolution kernel for 3x3 Gaussian Blur
    constexpr float convolution_mask_array_blur_3x3[] = {
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
    };
    // clang-format on
    constexpr int convolution_mask_width_blur_3x3 = 3;
    constexpr int convolution_mask_height_blur_3x3 = 3;
    constexpr int convolution_mask_length_blur_3x3 = convolution_mask_width_blur_3x3 * convolution_mask_height_blur_3x3;
    auto convolution_mask_blur_3x3 = sycl::malloc_device<float>(convolution_mask_length_blur_3x3, q);
    q.memcpy(convolution_mask_blur_3x3, convolution_mask_array_blur_3x3, convolution_mask_length_blur_3x3 * sizeof(decltype(convolution_mask_array_blur_3x3)));
    auto convolution_kernel_blur_3x3 = vn::ConvolutionKernel<decltype(in), decltype(out), decltype(convolution_mask_blur_3x3), float, uint8_t>(channels, in, out, convolution_mask_blur_3x3, convolution_mask_width_blur_3x3, convolution_mask_height_blur_3x3);
    auto convolution_blur_3x3 = [&in, &out, &q, &bidimensional_shape, &convolution_kernel_blur_3x3] {
        q.parallel_for(bidimensional_shape, convolution_kernel_blur_3x3).wait_and_throw();
    };
    functions.push_back({ "Image Convolution (Gaussian Blur 3x3 Kernel)", "convolution-blur-3", true, convolution_blur_3x3 });

    // clang-format off
    // Convolution kernel for 5x5 Gaussian Blur
    constexpr float convolution_mask_array_blur_5x5[] = {
        1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f,
        4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
        6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f,
        4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
        1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f
    };
    // clang-format on
    constexpr int convolution_mask_width_blur_5x5 = 5;
    constexpr int convolution_mask_height_blur_5x5 = 5;
    constexpr int convolution_mask_length_blur_5x5 = convolution_mask_width_blur_5x5 * convolution_mask_height_blur_5x5;
    auto convolution_mask_blur_5x5 = sycl::malloc_device<float>(convolution_mask_length_blur_5x5, q);
    q.memcpy(convolution_mask_blur_5x5, convolution_mask_array_blur_5x5, convolution_mask_length_blur_5x5 * sizeof(decltype(convolution_mask_array_blur_5x5)));
    auto convolution_kernel_blur_5x5 = vn::ConvolutionKernel<decltype(in), decltype(out), decltype(convolution_mask_blur_5x5), float, uint8_t>(channels, in, out, convolution_mask_blur_5x5, convolution_mask_width_blur_5x5, convolution_mask_height_blur_5x5);
    auto convolution_blur_5x5 = [&in, &out, &q, &bidimensional_shape, &convolution_kernel_blur_5x5] {
        q.parallel_for(bidimensional_shape, convolution_kernel_blur_5x5).wait_and_throw();
    };
    functions.push_back({ "Image Convolution (Gaussian Blur 5x5 Kernel)", "convolution-blur-5", true, convolution_blur_5x5 });

    // Direct Gaussian Blur 3x3 Kernel
    auto gaussian_blur_3x3_kernel = vn::GaussianBlur3X3Kernel<decltype(in), decltype(out), uint8_t>(channels, in, out);
    auto gaussian_blur_3x3 = [&in, &out, &q, &bidimensional_shape, &gaussian_blur_3x3_kernel] {
        q.parallel_for(bidimensional_shape, gaussian_blur_3x3_kernel).wait_and_throw();
    };
    functions.push_back({ "Image Gaussian Blurring (3x3 Kernel)", "blur-3", true, gaussian_blur_3x3 });

    // Perform every benchmark
    for (auto& [title, prefix, save, func] : functions) {
        double delta_once, delta_total;
        {
            auto start = ch::high_resolution_clock::now();
            func();
            auto end = ch::high_resolution_clock::now();
            delta_once = ch::duration_cast<ch::microseconds>(end - start).count() * 0.001;
        }
        {
            auto start = ch::high_resolution_clock::now();
            for (size_t i = 0; i < rounds; ++i) func();
            auto end = ch::high_resolution_clock::now();
            delta_total = ch::duration_cast<ch::microseconds>(end - start).count() * 0.001;
        }
        std::cout << title << ": " << delta_once << "ms (once) | " << delta_total << "ms (" << rounds << " times)" << std::endl;
        if (save) save_func((outpath.generic_string() + prefix + "-" + inpath.filename().generic_string()).c_str());
    }

    // Free all elements
    sycl::free(in, q);
    sycl::free(out, q);
    sycl::free(erode_mask, q);
    sycl::free(dilate_mask, q);
    sycl::free(convolution_mask_blur_3x3, q);
    sycl::free(convolution_mask_blur_5x5, q);

    return 0;
}
