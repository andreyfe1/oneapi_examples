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

    using allocator_t = sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    allocator_t allocator(q);

    std::vector<int, allocator_t> usm_vec_1(n, allocator);
    std::vector<int, allocator_t> usm_vec_2(n, allocator);
    std::vector<int, allocator_t> usm_vec_3(n, allocator);
    int* usm_ptr_1 = usm_vec_1.data();
    int* usm_ptr_2 = usm_vec_2.data();
    int* usm_ptr_3 = usm_vec_3.data();

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
