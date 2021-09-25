#include <vector>
#include <CL/sycl.hpp>

bool check(int* usm_ptr, const int n)
{
    for (int id = 1; id < n; ++id)
    {
        if (usm_ptr[id] != 42 + id)
        {
            std::cout << "id == " << id << ", expected = " << 42 + id << ", got = " << usm_ptr[id] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    const int n = 10'000'000; // should be run on CPU to reproduce failure
    sycl::queue q{sycl::property::queue::in_order{}};
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    auto usm_deleter = [q](int* ptr) { sycl::free(ptr, q); };
    std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_1(sycl::malloc_shared<int>(n, q), usm_deleter);
    std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_2(sycl::malloc_shared<int>(n, q), usm_deleter);
    std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_3(sycl::malloc_shared<int>(n, q), usm_deleter);
    int* usm_ptr_1 = usm_uptr_1.get();
    int* usm_ptr_2 = usm_uptr_2.get();
    int* usm_ptr_3 = usm_uptr_3.get();

    q.fill(usm_ptr_1, 42, n);

    q.parallel_for(sycl::range<>(n), [=](auto id) { usm_ptr_2[id] = id; });

    sycl::event event_3 =
        q.parallel_for(sycl::range<>(n), [=](auto id) { usm_ptr_3[id] = usm_ptr_1[id] + usm_ptr_2[id]; });
    event_3.wait();

    if (check(usm_ptr_3, n))
        std::cout << "passed" << std::endl;
    else
        std::cout << "failed" << std::endl;
    return 0;
}
