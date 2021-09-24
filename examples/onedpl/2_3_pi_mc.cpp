#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/random>
#include <CL/sycl.hpp>

int main()
{
    sycl::queue q;
    const int n = 10'000'000;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    int sum = std::count_if(
        dpl::execution::make_device_policy(q),
        dpl::counting_iterator<int>(0),
        dpl::counting_iterator<int>(n),
        [=](int id){
            dpl::minstd_rand engine(/*seed*/ 7777, /*offset*/ 2 * id);
            dpl::uniform_real_distribution<double> distr(0.0, 1.0);

            double x = distr(engine);
            double y = distr(engine);
            return x * x + y * y <= 1.0;
    });

    double estimated_pi = 4.0 * (static_cast<double>(sum) / n);
    std::cout << estimated_pi << std::endl;

    return 0;
}
