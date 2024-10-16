#include <iostream>
#include <sycl/sycl.hpp>

const std::string secret{
    "Ifmmp-!xpsme\"\012J(n!tpssz-!Ebwf/!"
    "J(n!bgsbje!J!dbo(u!ep!uibu/!.!IBM\01"
};

const auto sz = secret.size();

int main() {
    auto q = sycl::queue{};
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    auto result = sycl::malloc_shared<char>(sz, q);
    std::memcpy(result, secret.data(), sz);

    q.parallel_for(sz, [=](auto& i) {
         result[i] -= 1;
     }).wait();

    std::cout << result << std::endl;
    sycl::free(result, q);

    return 0;
}
