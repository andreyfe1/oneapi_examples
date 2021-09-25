#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <CL/sycl.hpp>

int main()
{
    const int n = 1'000'000;
    sycl::queue q;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    auto usm_deleter = [q](int* ptr) { sycl::free(ptr, q); };
    std::unique_ptr<int, decltype(usm_deleter)> usm_uptr(sycl::malloc_shared<int>(n, q), usm_deleter);
    int* usm_ptr = usm_uptr.get();

    auto device_policy = dpl::execution::make_device_policy(q);
    std::fill_n(device_policy, usm_ptr, n, 42);

    // int reduced_val = std::accumulate(usm_ptr_1, usm_ptr_1 + n, 0);
    // int reduced_val = std::reduce(dpl::execution::unseq, usm_ptr_1, usm_ptr_1 + n, 0);
    int reduced_val = std::reduce(device_policy, usm_ptr, usm_ptr + n, 0);

    if (reduced_val == 42 * n)
        std::cout << "passed" << std::endl;
    else
        std::cout << "failed" << std::endl;
    return 0;
}
