#include <CL/sycl.hpp>

bool check(int* usm_ptr, const int n)
{
    for (int id = 0; id < n; ++id)
    {
        if (usm_ptr[id] != id)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    const int n = 1'000'000;
    sycl::queue q;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    int* usm_ptr = sycl::malloc_shared<int>(n, q);

    q.parallel_for(sycl::range<>(n), [=](auto id) { usm_ptr[id] = id; });
    q.wait();

    if (check(usm_ptr, n))
        std::cout << "passed" << std::endl;
    else
        std::cout << "failed" << std::endl;

    sycl::free(usm_ptr, q);
    return 0;
}
