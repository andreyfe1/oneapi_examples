#include <vector>
#include <CL/sycl.hpp>

bool check(int* usm_ptr, const int n){
    for(int id = 1; id < n; ++id){
        if(usm_ptr[id] != 42 + id)
        {
            std::cout<<"id == "<<id<<", expected = "<<42 + id<<", got = "<<usm_ptr[id]<<std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    sycl::queue q;
    const int n = 10'000'000; // should be run on CPU to reproduce failure
    int* usm_ptr_1 = sycl::malloc_shared<int>(n, q);
    int* usm_ptr_2 = sycl::malloc_shared<int>(n, q);
    int* usm_ptr_3 = sycl::malloc_shared<int>(n, q);
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    sycl::event event_1 =
        q.fill(usm_ptr_1, int(42), n);

    sycl::event event_2 =
        q.parallel_for(sycl::range<>(n),
            [=](auto id) {
                usm_ptr_2[id] = id;
            });

    sycl::event event_3 =
        q.parallel_for(sycl::range<>(n),
            std::vector<sycl::event>{event_1, event_2},
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

    // sycl::event event_1 = q.submit([&](sycl::handler& cgh) {
    //     cgh.fill(usm_ptr_1, 42, n);
    // });

    // sycl::event event_2 = q.submit([&](sycl::handler& cgh) {
    //     cgh.parallel_for(sycl::range<>(n),
    //         [=](auto id) {
    //             usm_ptr_2[id] = id;
    //         }
    //     );
    // });

    // sycl::event event_3 = q.submit([&](sycl::handler& cgh) {
    //     cgh.depends_on(std::vector<sycl::event>{event_1, event_2});
    //     cgh.parallel_for(sycl::range<>(n),
    //         [=](auto id) {
    //             usm_ptr_3[id] = usm_ptr_1[id] + usm_ptr_2[id];
    //         }
    //     );
    // });
    // event_3.wait();
