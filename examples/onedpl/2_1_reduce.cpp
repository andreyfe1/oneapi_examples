#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <CL/sycl.hpp>

int main()
{
    sycl::queue q;
    const int n = 1'000'000;
    int* usm_ptr_1 = sycl::malloc_shared<int>(n, q);
    int* reduced_val = sycl::malloc_host<int>(1, q);

    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    auto device_policy = dpl::execution::make_device_policy(q);
    std::fill_n(device_policy, usm_ptr_1, n, 42);

    // *reduced_val = std::accumulate(usm_ptr_1, usm_ptr_1 + n, 0);
    // *reduced_val = std::reduce(dpl::execution::unseq, usm_ptr_1, usm_ptr_1 + n, 0);
    *reduced_val = std::reduce(device_policy, usm_ptr_1, usm_ptr_1 + n, 0);

    if(*reduced_val == 42 * n)
        std::cout << "passed" << std::endl;
    else
        std::cout << "failed" << std::endl;

    sycl::free(usm_ptr_1, q);
    sycl::free(reduced_val, q);
    return 0;
}
