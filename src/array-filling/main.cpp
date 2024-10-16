#include <array>
#include <iostream>
#include <sycl/sycl.hpp>

int main() {
    constexpr int size = 16;
    std::array<int, size> data;

    sycl::buffer B{ data };
    sycl::queue q{};

    // Select any device for this queue
    std::cout << "Device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    q.submit([&](sycl::handler& h) {
        sycl::accessor acc{ B, h };

        h.parallel_for(size, [=](auto& idx) {
            acc[idx] = idx;
        });
    });

    auto acc = sycl::host_accessor{ B };

    for (auto n : acc) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    return 0;
}
