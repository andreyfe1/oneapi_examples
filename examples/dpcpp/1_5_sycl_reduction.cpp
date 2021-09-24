#include <CL/sycl.hpp>

int main()
{
    sycl::queue q;
    const int n = 1 << 20;
    int* usm_ptr_1 = sycl::malloc_shared<int>(n, q);
    int* reduced_val = sycl::malloc_host<int>(1, q);

    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    sycl::event event_1 =
        q.fill(usm_ptr_1, 42, n);
    event_1.wait();

    auto reductor = sycl::reduction(reduced_val, 0, std::plus<int>{});

    auto event_2 = q.parallel_for(sycl::nd_range<>{{n}, {64}},
        reductor,
        [=](auto nd_id, auto &sum) { sum += static_cast<int>(usm_ptr_1[nd_id.get_global_id()]); }
    );
    event_2.wait();

    if(*reduced_val == 42 * n)
        std::cout << "passed" << std::endl;
    else
        std::cout << "failed" << std::endl;

    sycl::free(usm_ptr_1, q);
    sycl::free(reduced_val, q);
    return 0;
}
