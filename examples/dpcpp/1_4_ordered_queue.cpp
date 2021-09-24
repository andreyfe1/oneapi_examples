#include <vector>
#include <CL/sycl.hpp>

bool check(int* usm_ptr, const int n){
    for(int id = 0; id < n; ++id){
        if(usm_ptr[id] != 42 + id){
            return false;
        }
    }
    return true;
}

int main()
{
    sycl::queue q{sycl::property::queue::in_order{}};
    const int n = 100'000'000;
    int* usm_ptr_1 = sycl::malloc_shared<int>(n, q);
    int* usm_ptr_2 = sycl::malloc_shared<int>(n, q);
    int* usm_ptr_3 = sycl::malloc_shared<int>(n, q);
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    q.fill(usm_ptr_1, 42, n);

    q.parallel_for(sycl::range<>(n),
        [=](auto id) {
            usm_ptr_2[id] = id;
        });

    sycl::event event_3 = q.parallel_for(sycl::range<>(n),
        [=](auto id) {
            usm_ptr_3[id] = usm_ptr_1[id] + usm_ptr_2[id];
        });
    event_3.wait();

    if(check(usm_ptr_3, n))
        std::cout << "passed" << std::endl;
    else
        std::cout << "failed" << std::endl;

    sycl::free(usm_ptr_1, q);
    sycl::free(usm_ptr_2, q);
    sycl::free(usm_ptr_3, q);
    return 0;
}
