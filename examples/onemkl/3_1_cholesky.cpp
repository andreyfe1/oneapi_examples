#include <iostream>
#include <vector>

#include <oneapi/mkl.hpp>

template <typename T, typename Allocator>
void initialize(sycl::queue& q, std::vector<T, Allocator>& A, std::vector<T, Allocator>& B, const std::int64_t n)
{
    T* A_ptr = A.data();
    T* B_ptr = B.data();
    q.parallel_for(sycl::range<>(n * n),
                   [=](auto id)
                   {
                       auto j = id / n;
                       auto i = id % n;
                       A_ptr[j * n + i] = (i == j) ? 2 * i + 1 : static_cast<T>(i + j) / static_cast<T>(n);

                       if (j == 0)
                           B_ptr[i] = i;
                   })
        .wait();
}

int main()
{
    using T = double;
    namespace mkl_lapack = oneapi::mkl::lapack;
    namespace mkl_blas = oneapi::mkl::blas;

    sycl::queue q;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Matrix sizes and leading dimensions
    std::int64_t n = 10'000;
    std::int64_t lda = n, ldb = n;
    oneapi::mkl::uplo mkl_lower = oneapi::mkl::uplo::lower;

    using allocator_t = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
    allocator_t allocator(q);

    // Allocate matrices
    std::vector<T, allocator_t> A(n * n, allocator);
    std::vector<T, allocator_t> B(n, allocator);
    std::vector<T, allocator_t> result(1, allocator);

    initialize(q, A, B, n);
    // store local copies
    std::vector<T, allocator_t> A_copy(A.begin(), A.end(), allocator);
    std::vector<T, allocator_t> B_copy(B.begin(), B.end(), allocator);

    // Allocate scratchpads for calculations
    std::int64_t potrf_temp_size = mkl_lapack::potrf_scratchpad_size<T>(q, mkl_lower, n, lda);
    std::int64_t potrs_temp_size = mkl_lapack::potrs_scratchpad_size<T>(q, mkl_lower, n, 1, lda, ldb);
    std::vector<T, allocator_t> scratchpad(std::max(potrf_temp_size, potrs_temp_size), allocator);

    // factorization
    sycl::event potrf_event =
        mkl_lapack::potrf(q, mkl_lower, n, /*result*/ A.data(), lda, scratchpad.data(), potrf_temp_size);

    // solver
    sycl::event potrs_event = mkl_lapack::potrs(q, mkl_lower, n, 1, A.data(), lda, /*result*/ B.data(), ldb,
                                                scratchpad.data(), potrs_temp_size, {potrf_event});

    // check
    sycl::event symv_event = mkl_blas::symv(q, mkl_lower, n, 1.0, A_copy.data(), lda, B.data(), 1, -1.0,
                                            /*result*/ B_copy.data(), 1, {potrs_event});
    sycl::event nrm2_event = mkl_blas::nrm2(q, n, B_copy.data(), 1, result.data(), {symv_event});
    nrm2_event.wait();

    std::cout << "norm = " << result[0] << std::endl;
    return 0;
}
