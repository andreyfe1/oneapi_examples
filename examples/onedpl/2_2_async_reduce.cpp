#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>

int main()
{
    sycl::queue q;
    const int n = 1'000'000;
    int* usm_ptr_1 = sycl::malloc_shared<int>(n, q);
    int* reduced_val = sycl::malloc_host<int>(1, q);

    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    auto device_policy = dpl::execution::make_device_policy(q);
    auto fill_future = dpl::experimental::fill_async(device_policy, usm_ptr_1, usm_ptr_1 + n, 42);

    auto reduce_future = dpl::experimental::reduce_async(device_policy, usm_ptr_1, usm_ptr_1 + n, fill_future);
    *reduced_val = reduce_future.get();

    if(*reduced_val == 42 * n)
        std::cout << "passed" << std::endl;
    else
        std::cout << "failed" << std::endl;

    sycl::free(usm_ptr_1, q);
    sycl::free(reduced_val, q);
    return 0;
}
