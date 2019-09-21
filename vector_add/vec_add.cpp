#include <iostream>
#include <vector>
#include <chrono>

#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

#if CPU
sycl::queue q{cl::sycl::cpu_selector{}};
#else
sycl::queue q{cl::sycl::gpu_selector{}};
#endif

template<typename Buffer, typename T>
void sycl_fill(Buffer buf, const T& value) {
    q.submit([&](sycl::handler& cgh) {
        auto acc = buf.template get_access<sycl::access::mode::discard_write>(cgh);

#if USE_FILL
        cgh.fill(acc, value * value + value);
#else
        cgh.parallel_for<class fill>(
            sycl::range<1>(buf.get_count()),
            [=](sycl::item<1> it) {
              acc[it] = value * value + value;
        });
    });
#endif
}

template<typename Buffer>
void sycl_add(Buffer buf1, Buffer buf2) {
    q.submit([&](sycl::handler& cgh) {
        auto acc1 = buf1.template get_access<sycl::access::mode::read_write>(cgh);
        auto acc2 = buf2.template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<class add>(
            sycl::range<1>(buf1.get_count()),
            [=](sycl::item<1> it) {
              acc1[it] += acc2[it];
        });
    });
}

template<typename Buffer, typename T>
bool are_results_correct(Buffer buf1, Buffer buf2, const T& value1, const T& value2){
    auto host_acc1 = buf1.template get_access<sycl::access::mode::read>();
    auto host_acc2 = buf2.template get_access<sycl::access::mode::read>();
    for (int i = 0; i < buf1.get_count(); ++i) {
        if (host_acc1[i] != value1 * value1 + value1 && host_acc2[i] != value2 * value2 + value2) {
            std::cout << "test failed. i  = " << i << ", acc1 = " << host_acc1[i] << ", acc2[i] = " << host_acc2[i] << std::endl;
            return false;
        }
    }
    return true;
}

using namespace std::chrono;
int main()
{
    const size_t n = 1e8;
    using T = int;
    std::cout << "Running on " << q.get_device().template get_info<sycl::info::device::name>() << std::endl;

    sycl::buffer<T, 1> buf1(n);
    sycl::buffer<T, 1> buf2(n);

    for (int iter = 0; iter < 10; iter++) {

        T value1 = 3 + iter;
        T value2 = 5 + iter;

        auto start = high_resolution_clock::now();

        sycl_fill(buf1, value1);
        sycl_fill(buf2, value2);

        sycl_add(buf1, buf2);
        q.wait();

        auto end = high_resolution_clock::now();
        duration<double> diff = end - start;
        std::cout << "calculation time is " << diff.count() << std::endl;

        //check results
        if (iter == 0) {
            if(!are_results_correct(buf1, buf2, value1, value2)){
                std::cout << "test failed" << std::endl;
                return 1;
            }
        }
    }
    std::cout << "test passed" << std::endl;
    return 0;
}
