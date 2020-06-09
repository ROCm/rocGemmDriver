#ifndef _UTILITY_
#define _UTILITY_

#define ROCM_USE_FLOAT16

#include "rocblas.h"
#include <cmath>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <vector>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void)
{
    hipDeviceSynchronize();
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
};

/* ============================================================================================ */
/*  device query and print out their ID and name; return number of compute-capable devices. */
rocblas_int query_device_property()
{
    int            device_count;
    rocblas_status status = (rocblas_status)hipGetDeviceCount(&device_count);
    if(status != rocblas_status_success)
    {
        printf("Query device error: cannot get device count \n");
        return -1;
    }
    else
    {
        printf("Query device success: there are %d devices \n", device_count);
    }

    for(rocblas_int i = 0;; i++)
    {
        puts("-------------------------------------------------------------------------------");

        if(i >= device_count)
        {
            break;
        }

        hipDeviceProp_t props;
        rocblas_status  status = (rocblas_status)hipGetDeviceProperties(&props, i);
        if(status != rocblas_status_success)
        {
            printf("Query device error: cannot get device ID %d's property\n", i);
        }
        else
        {
            printf("Device ID %d : %s\n", i, props.name);
            printf("with %3.1f GB memory, clock rate %dMHz @ computing capability %d.%d \n",
                   props.totalGlobalMem / 1e9,
                   (int)(props.clockRate / 1000),
                   props.major,
                   props.minor);
            printf(
                "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                props.maxGridSize[0],
                props.sharedMemPerBlock / 1e3,
                props.maxThreadsPerBlock,
                props.warpSize);
        }
    }
    
    std::cout << std::endl;
    return device_count;
}

/*  set current device to device_id */
void set_device(rocblas_int device_id)
{
    rocblas_status status = (rocblas_status)hipSetDevice(device_id);
    if(status != rocblas_status_success)
    {
        printf("Set device error: cannot set device ID %d, there may not be such device ID\n",
               (int)device_id);
    }
}

inline const char* rocblas_status_to_string(rocblas_status status)
{
    switch(status)
    {
    case rocblas_status_success:
        return "rocblas_status_success";
    case rocblas_status_invalid_handle:
        return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
        return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
        return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
        return "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
        return "rocblas_status_memory_error";
    case rocblas_status_internal_error:
        return "rocblas_status_internal_error";
    default:
        return "<undefined rocblas_status value>";
    }
}

inline void rocblas_expect_status(rocblas_status status, rocblas_status expect)
{
    if(status != expect)
    {
        std::cerr << "rocBLAS status error: Expected " << rocblas_status_to_string(expect)
                  << ", received " << rocblas_status_to_string(status) << std::endl;
        if(expect == rocblas_status_success)
            exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP_ERROR(ERROR)                    \
    do                                            \
    {                                             \
        auto error = ERROR;                       \
        if(error != hipSuccess)                   \
        {                                         \
            fprintf(stderr,                       \
                    "error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error),     \
                    error,                        \
                    __FILE__,                     \
                    __LINE__);                    \
            exit(EXIT_FAILURE);                   \
        }                                         \
    } while(0)

#define EXPECT_ROCBLAS_STATUS rocblas_expect_status

#define CHECK_ROCBLAS_ERROR2(STATUS) EXPECT_ROCBLAS_STATUS(STATUS, rocblas_status_success)
#define CHECK_ROCBLAS_ERROR(STATUS) CHECK_ROCBLAS_ERROR2(STATUS)

// gemm
template <typename T>
rocblas_status (*rocblas_gemm)(rocblas_handle    handle,
                               rocblas_operation transA,
                               rocblas_operation transB,
                               rocblas_int       m,
                               rocblas_int       n,
                               rocblas_int       k,
                               const T*          alpha,
                               const T*          A,
                               rocblas_int       lda,
                               const T*          B,
                               rocblas_int       ldb,
                               const T*          beta,
                               T*                C,
                               rocblas_int       ldc);

template <>
constexpr auto rocblas_gemm<rocblas_half> = rocblas_hgemm;

template <>
constexpr auto rocblas_gemm<float> = rocblas_sgemm;

template <>
constexpr auto rocblas_gemm<double> = rocblas_dgemm;

// gemm_strided_batched
template <typename T>
rocblas_status (*rocblas_gemm_strided_batched)(rocblas_handle    handle,
                                               rocblas_operation transA,
                                               rocblas_operation transB,
                                               rocblas_int       m,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const T*          alpha,
                                               const T*          A,
                                               rocblas_int       lda,
                                               rocblas_stride    bsa,
                                               const T*          B,
                                               rocblas_int       ldb,
                                               rocblas_stride    bsb,
                                               const T*          beta,
                                               T*                C,
                                               rocblas_int       ldc,
                                               rocblas_stride    bsc,
                                               rocblas_int       batch_count);

template <>
constexpr auto rocblas_gemm_strided_batched<rocblas_half> = rocblas_hgemm_strided_batched;

template <>
constexpr auto rocblas_gemm_strided_batched<float> = rocblas_sgemm_strided_batched;

template <>
constexpr auto rocblas_gemm_strided_batched<double> = rocblas_dgemm_strided_batched;

/* ============================================================================================ */
// Random number generator
using rocblas_rng_t = std::mt19937;
extern rocblas_rng_t rocblas_rng, rocblas_seed;

rocblas_rng_t rocblas_rng, rocblas_seed;

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void rocblas_seedrand()
{
    rocblas_rng = rocblas_seed;
}

/* ============================================================================================ */
/*! \brief  Random number generator which generates NaN values */
class rocblas_nan_rng
{
    // Generate random NaN values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union u_t
        {
            u_t() {}
            UINT_T u;
            T      fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(rocblas_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(rocblas_rng);
    }

    // Random NaN double
    explicit operator double()
    {
        return random_nan_data<double, uint64_t, 52, 11>();
    }

    // Random NaN float
    explicit operator float()
    {
        return random_nan_data<float, uint32_t, 23, 8>();
    }

    // Random NaN half (non-template rocblas_half takes precedence over integer template above)
    explicit operator rocblas_half()
    {
        return random_nan_data<rocblas_half, uint16_t, 10, 5>();
    }

    // Random NaN bfloat16
    explicit operator rocblas_bfloat16()
    {
        return random_nan_data<rocblas_bfloat16, uint16_t, 7, 8>();
    }

    explicit operator rocblas_float_complex()
    {
        return {float(*this), float(*this)};
    }

    explicit operator rocblas_double_complex()
    {
        return {double(*this), double(*this)};
    }
};

/* ============================================================================================ */
// Helper function to truncate float to bfloat16

inline __host__ rocblas_bfloat16 float_to_bfloat16_truncate(float val)
{
    union
    {
        float    fp32;
        uint32_t int32;
    } u = {val};
    rocblas_bfloat16 ret;
    ret.data = uint16_t(u.int32 >> 16);
    if((u.int32 & 0x7fff0000) == 0x7f800000 && u.int32 & 0xffff)
        ret.data |= 1; // Preserve signaling NaN
    return ret;
}

/* ============================================================================================ */
/*! \brief  returns true if value is NaN */

template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
inline bool rocblas_isnan(T)
{
    return false;
}

template <typename T,
          typename std::enable_if<!std::is_integral<T>{} && !is_complex<T>, int>::type = 0>
inline bool rocblas_isnan(T arg)
{
    return std::isnan(arg);
}

template <typename T, typename std::enable_if<is_complex<T>, int>::type = 0>
inline bool rocblas_isnan(const T& arg)
{
    return rocblas_isnan(std::real(arg)) || rocblas_isnan(std::imag(arg));
}

inline bool rocblas_isnan(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t     data;
    } x = {arg};
    return (~x.data & 0x7c00) == 0 && (x.data & 0x3ff) != 0;
}

/* ============================================================================================ */
/*! \brief negate a value */

template <class T>
inline T negate(T x)
{
    return -x;
}

template <>
inline rocblas_half negate(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t     data;
    } x = {arg};

    x.data ^= 0x8000;
    return x.fp;
}

template <>
inline rocblas_bfloat16 negate(rocblas_bfloat16 x)
{
    x.data ^= 0x8000;
    return x;
}

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
inline T random_generator()
{
    return std::uniform_int_distribution<int>(1, 10)(rocblas_rng);
}

// for rocblas_float_complex, generate two random ints (same behaviour as for floats)
template <>
inline rocblas_float_complex random_generator<rocblas_float_complex>()
{
    return {float(std::uniform_int_distribution<int>(1, 10)(rocblas_rng)),
            float(std::uniform_int_distribution<int>(1, 10)(rocblas_rng))};
};

// for rocblas_double_complex, generate two random ints (same behaviour as for doubles)
template <>
inline rocblas_double_complex random_generator<rocblas_double_complex>()
{
    return {double(std::uniform_int_distribution<int>(1, 10)(rocblas_rng)),
            double(std::uniform_int_distribution<int>(1, 10)(rocblas_rng))};
};

// for rocblas_half, generate float, and convert to rocblas_half
/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
inline rocblas_half random_generator<rocblas_half>()
{
    return float_to_bfloat16_truncate(std::uniform_int_distribution<int>(-2, 2)(rocblas_rng));
};

// for rocblas_bfloat16, generate float, and convert to rocblas_bfloat16
/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
inline rocblas_bfloat16 random_generator<rocblas_bfloat16>()
{
    return rocblas_bfloat16(std::uniform_int_distribution<int>(-2, 2)(rocblas_rng));
};

/*! \brief  generate a random number in range [1,2,3] */
template <>
inline int8_t random_generator<int8_t>()
{
    return std::uniform_int_distribution<int8_t>(1, 3)(rocblas_rng);
};

/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
inline T random_hpl_generator()
{
    return std::uniform_real_distribution<double>(-0.5, 0.5)(rocblas_rng);
}

template <>
inline rocblas_bfloat16 random_hpl_generator()
{
    return rocblas_bfloat16(std::uniform_real_distribution<double>(-0.5, 0.5)(rocblas_rng));
}

/* ============================================================================================ */
/*! \brief  base-class to allocate/deallocate device memory */
template <typename T, size_t PAD, typename U>
class d_vector
{
private:
    size_t size, bytes;

public:
    inline size_t nmemb() const noexcept
    {
        return size;
    }

#ifdef GOOGLE_TEST
    U guard[PAD];
    d_vector(size_t s)
        : size(s)
        , bytes((s + PAD * 2) * sizeof(T))
    {
        // Initialize guard with random data
        if(PAD > 0)
        {
            rocblas_init_nan(guard, PAD);
        }
    }
#else
    d_vector(size_t s)
        : size(s)
        , bytes(s ? s * sizeof(T) : sizeof(T))
    {
    }
#endif

    T* device_vector_setup()
    {
        T* d;
        if((hipMalloc)(&d, bytes) != hipSuccess)
        {
            static char* lc = setlocale(LC_NUMERIC, "");
            fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", bytes, bytes >> 30);

            d = nullptr;
        }
#ifdef GOOGLE_TEST
        else
        {
            if(PAD > 0)
            {
                // Copy guard to device memory before allocated memory
                hipMemcpy(d, guard, sizeof(guard), hipMemcpyHostToDevice);

                // Point to allocated block
                d += PAD;

                // Copy guard to device memory after allocated memory
                hipMemcpy(d + size, guard, sizeof(guard), hipMemcpyHostToDevice);
            }
        }
#endif
        return d;
    }

    void device_vector_check(T* d)
    {
#ifdef GOOGLE_TEST
        if(PAD > 0)
        {
            U host[PAD];

            // Copy device memory after allocated memory to host
            hipMemcpy(host, d + this->size, sizeof(guard), hipMemcpyDeviceToHost);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

            // Point to guard before allocated memory
            d -= PAD;

            // Copy device memory after allocated memory to host
            hipMemcpy(host, d, sizeof(guard), hipMemcpyDeviceToHost);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);
        }
#endif
    }

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
#ifdef GOOGLE_TEST
            if(PAD > 0)
            {
                U host[PAD];

                // Copy device memory after allocated memory to host
                hipMemcpy(host, d + this->size, sizeof(guard), hipMemcpyDeviceToHost);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

                // Point to guard before allocated memory
                d -= PAD;

                // Copy device memory after allocated memory to host
                hipMemcpy(host, d, sizeof(guard), hipMemcpyDeviceToHost);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);
            }
#endif
            // Free device memory
            CHECK_HIP_ERROR((hipFree)(d));
        }
    }
};

//
// Local declaration of the host vector.
//
template <typename T>
class host_vector;

//!
//! @brief pseudo-vector subclass which uses device memory
//!
template <typename T, size_t PAD = 4096, typename U = T>
class device_vector : private d_vector<T, PAD, U>
{

public:
    //!
    //! @brief Disallow copying.
    //!
    device_vector(const device_vector&) = delete;

    //!
    //! @brief Disallow assigning
    //!
    device_vector& operator=(const device_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param n The length of the vector.
    //! @param inc The increment.
    //! @remark Must wrap constructor and destructor in functions to allow Google Test macros to work
    //!
    explicit device_vector(rocblas_int n, rocblas_int inc)
        : d_vector<T, PAD, U>(n * std::abs(inc))
        , m_n(n)
        , m_inc(inc)
    {
        this->m_data = this->device_vector_setup();
    }

    //!
    //! @brief Constructor (kept for backward compatibility)
    //! @param s the size.
    //! @remark Must wrap constructor and destructor in functions to allow Google Test macros to work
    //!
    explicit device_vector(size_t s)
        : d_vector<T, PAD, U>(s)
        , m_n(s)
        , m_inc(1)
    {
        this->m_data = this->device_vector_setup();
    }

    //!
    //! @brief Destructor.
    //!
    ~device_vector()
    {
        this->device_vector_teardown(this->m_data);
        this->m_data = nullptr;
    }

    //!
    //! @brief Returns the length of the vector.
    //!
    rocblas_int n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the increment of the vector.
    //!
    rocblas_int inc() const
    {
        return this->m_inc;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    rocblas_int batch_count() const
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    rocblas_stride stride() const
    {
        return 0;
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected.
    //!
    operator T*()
    {
        return this->m_data;
    }

    //!
    //! @brief Decay into constant pointer wherever pointer is expected.
    //!
    operator const T*() const
    {
        return this->m_data;
    }

    //!
    //! @brief Tell whether malloc failed.
    //!
    explicit operator bool() const
    {
        return nullptr != this->m_data;
    }

    //!
    //! @brief Transfer data from a host vector.
    //! @param that The host vector.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const host_vector<T>& that)
    {
        return hipMemcpy(
            this->m_data, (const T*)that, this->nmemb() * sizeof(T), hipMemcpyHostToDevice);
    }

    hipError_t memcheck() const
    {
        return ((bool)*this) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t      m_size{};
    rocblas_int m_n{};
    rocblas_int m_inc{};
    T*          m_data{};
};

/* ============================================================================================ */
//!
//! @brief  Pseudo-vector subclass which uses host memory.
//!
template <typename T>
struct host_vector : std::vector<T>
{
    // Inherit constructors
    using std::vector<T>::vector;

    //!
    //! @brief Constructor.
    //!
    host_vector(rocblas_int n, rocblas_int inc)
        : std::vector<T>(n * std::abs(inc))
        , m_n(n)
        , m_inc(inc)
    {
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected
    //!
    operator T*()
    {
        return this->data();
    }

    //!
    //! @brief Decay into constant pointer wherever constant pointer is expected
    //!
    operator const T*() const
    {
        return this->data();
    }

    //!
    //! @brief Transfer from a device vector.
    //! @param  that That device vector.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_vector<T>& that)
    {
        return hipMemcpy(
            this->data(), (const T*)that, sizeof(T) * this->size(), hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Returns the length of the vector.
    //!
    rocblas_int n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the increment of the vector.
    //!
    rocblas_int inc() const
    {
        return this->m_inc;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    rocblas_int batch_count() const
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    rocblas_stride stride() const
    {
        return 0;
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        return (nullptr != (const T*)this) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    rocblas_int m_n{};
    rocblas_int m_inc{};
};

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values
template <typename T>
inline void rocblas_init(
    std::vector<T>& A, size_t M, size_t N, size_t lda, rocblas_stride stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_generator<T>();
}

template <typename T>
inline void rocblas_init_sin(
    std::vector<T>& A, size_t M, size_t N, size_t lda, rocblas_stride stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(sin(i + j * lda + i_batch * stride));
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
inline void rocblas_init_alternating_sign(
    std::vector<T>& A, size_t M, size_t N, size_t lda, rocblas_stride stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                auto value                        = random_generator<T>();
                A[i + j * lda + i_batch * stride] = (i ^ j) & 1 ? value : negate(value);
            }
}

template <typename T>
inline void rocblas_init_cos(
    std::vector<T>& A, size_t M, size_t N, size_t lda, rocblas_stride stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(cos(i + j * lda + i_batch * stride));
}

/*! \brief  symmetric matrix initialization: */
// for real matrix only
template <typename T>
inline void rocblas_init_symmetric(std::vector<T>& A, size_t N, size_t lda)
{
    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j <= i; ++j)
        {
            auto value = random_generator<T>();
            // Warning: It's undefined behavior to assign to the
            // same array element twice in same sequence point (i==j)
            A[j + i * lda] = value;
            A[i + j * lda] = value;
        }
}

/*! \brief  hermitian matrix initialization: */
// for complex matrix only, the real/imag part would be initialized with the same value
// except the diagonal elment must be real
template <typename T>
inline void rocblas_init_hermitian(std::vector<T>& A, size_t N, size_t lda)
{
    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j <= i; ++j)
        {
            auto value     = random_generator<T>();
            A[j + i * lda] = value;
            value.y        = (i == j) ? 0 : negate(value.y);
            A[i + j * lda] = value;
        }
}

// Initialize vector with HPL-like random values
template <typename T>
inline void rocblas_init_hpl(
    std::vector<T>& A, size_t M, size_t N, size_t lda, rocblas_stride stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_hpl_generator<T>();
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
inline void rocblas_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocblas_nan_rng());
}

template <typename T>
inline void rocblas_init_nan(
    std::vector<T>& A, size_t M, size_t N, size_t lda, rocblas_stride stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_nan_rng());
}

/* ============================================================================================ */
/*! \brief  Packs strided_batched matricies into groups of 4 in N */

template <typename T>
inline void rocblas_packInt8(
    std::vector<T>& A, size_t M, size_t N, size_t batch_count, size_t lda, rocblas_stride stride_a)
{
    if(N % 4 != 0)
        std::cerr << "ERROR: dimension must be a multiple of 4 in order to pack" << std::endl;

    std::vector<T> temp(A);
    for(size_t count = 0; count < batch_count; count++)
        for(size_t colBase = 0; colBase < N; colBase += 4)
            for(size_t row = 0; row < lda; row++)
                for(size_t colOffset = 0; colOffset < 4; colOffset++)
                    A[(colBase * lda + 4 * row) + colOffset + (stride_a * count)]
                        = temp[(colBase + colOffset) * lda + row + (stride_a * count)];
}

/* ============================================================================================ */
/*! \brief  Packs matricies into groups of 4 in N */
template <typename T>
inline void rocblas_packInt8(std::vector<T>& A, size_t M, size_t N, size_t lda)
{
    /* Assumes original matrix provided in column major order, where N is a multiple of 4

        ---------- N ----------
   |  | 00 05 10 15 20 25 30 35      |00 05 10 15|20 25 30 35|
   |  | 01 06 11 16 21 26 31 36      |01 06 11 16|21 26 31 36|
   l  M 02 07 12 17 22 27 32 37  --> |02 07 12 17|22 27 32 37|
   d  | 03 08 13 18 23 28 33 38      |03 08 13 18|23 28 33 38|
   a  | 04 09 14 19 24 29 34 39      |04 09 14 19|24 29 34 39|
   |    ** ** ** ** ** ** ** **      |** ** ** **|** ** ** **|
   |    ** ** ** ** ** ** ** **      |** ** ** **|** ** ** **|

     Input :  00 01 02 03 04 ** ** 05   ...  38 39 ** **
     Output:  00 05 10 15 01 06 11 16   ...  ** ** ** **

   */

    //  call general code with batch_count = 1 and stride_a = 0
    rocblas_packInt8(A, M, N, 1, lda, 0);
}

/* \brief floating point counts of GEMM */
template <typename T>
constexpr double gemm_gflop_count(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (2.0 * m * n * k) / 1e9;
}

template <>
constexpr double
    gemm_gflop_count<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (8.0 * m * n * k) / 1e9;
}

template <>
constexpr double
    gemm_gflop_count<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (8.0 * m * n * k) / 1e9;
}

typedef enum rocblas_initialization_
{
    rocblas_initialization_random_int    = 111,
    rocblas_initialization_random_narrow = 222,
    rocblas_initialization_random_broad  = 333,
    rocblas_initialization_random_full   = 444,
    rocblas_initialization_trig_float    = 555,
    rocblas_initialization_hpl           = 666,
    rocblas_initialization_const         = 777,
    rocblas_initialization_file          = 888,
} rocblas_initialization;

/* ============================================================================================ */
/*  Convert rocblas constants to lapack char. */

constexpr auto rocblas2char_operation(rocblas_operation value)
{
    switch(value)
    {
    case rocblas_operation_none:
        return 'N';
    case rocblas_operation_transpose:
        return 'T';
    case rocblas_operation_conjugate_transpose:
        return 'C';
    }
    return '\0';
}

constexpr auto rocblas2char_fill(rocblas_fill value)
{
    switch(value)
    {
    case rocblas_fill_upper:
        return 'U';
    case rocblas_fill_lower:
        return 'L';
    case rocblas_fill_full:
        return 'F';
    }
    return '\0';
}

constexpr auto rocblas2char_diagonal(rocblas_diagonal value)
{
    switch(value)
    {
    case rocblas_diagonal_unit:
        return 'U';
    case rocblas_diagonal_non_unit:
        return 'N';
    }
    return '\0';
}

constexpr auto rocblas2char_side(rocblas_side value)
{
    switch(value)
    {
    case rocblas_side_left:
        return 'L';
    case rocblas_side_right:
        return 'R';
    case rocblas_side_both:
        return 'B';
    }
    return '\0';
}

// return precision string for rocblas_datatype
constexpr auto rocblas_datatype2string(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:
        return "f16_r";
    case rocblas_datatype_f32_r:
        return "f32_r";
    case rocblas_datatype_f64_r:
        return "f64_r";
    case rocblas_datatype_f16_c:
        return "f16_k";
    case rocblas_datatype_f32_c:
        return "f32_c";
    case rocblas_datatype_f64_c:
        return "f64_c";
    case rocblas_datatype_i8_r:
        return "i8_r";
    case rocblas_datatype_u8_r:
        return "u8_r";
    case rocblas_datatype_i32_r:
        return "i32_r";
    case rocblas_datatype_u32_r:
        return "u32_r";
    case rocblas_datatype_i8_c:
        return "i8_c";
    case rocblas_datatype_u8_c:
        return "u8_c";
    case rocblas_datatype_i32_c:
        return "i32_c";
    case rocblas_datatype_u32_c:
        return "u32_c";
    case rocblas_datatype_bf16_r:
        return "bf16_r";
    case rocblas_datatype_bf16_c:
        return "bf16_c";
    default:
        return "invalid";
    }
}

constexpr auto rocblas_initialization2string(rocblas_initialization init)
{
    switch(init)
    {
    case rocblas_initialization_random_int:
        return "rand_int";
    case rocblas_initialization_random_narrow:
        return "rand_narrow";
    case rocblas_initialization_random_broad:
        return "rand_broad";
    case rocblas_initialization_random_full:
        return "rand_full";
    case rocblas_initialization_trig_float:
        return "trig_float";
    case rocblas_initialization_hpl:
        return "hpl";
    case rocblas_initialization_const:
        return "const";
    case rocblas_initialization_file:
        return "file";
    default:
        return "invalid";
    }
}

/* ============================================================================================ */
/*  Convert lapack char constants to rocblas type. */

constexpr rocblas_operation char2rocblas_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n':
        return rocblas_operation_none;
    case 'T':
    case 't':
        return rocblas_operation_transpose;
    case 'C':
    case 'c':
        return rocblas_operation_conjugate_transpose;
    default:
        return static_cast<rocblas_operation>(-1);
    }
}

constexpr rocblas_fill char2rocblas_fill(char value)
{
    switch(value)
    {
    case 'U':
    case 'u':
        return rocblas_fill_upper;
    case 'L':
    case 'l':
        return rocblas_fill_lower;
    default:
        return static_cast<rocblas_fill>(-1);
    }
}

constexpr rocblas_diagonal char2rocblas_diagonal(char value)
{
    switch(value)
    {
    case 'U':
    case 'u':
        return rocblas_diagonal_unit;
    case 'N':
    case 'n':
        return rocblas_diagonal_non_unit;
    default:
        return static_cast<rocblas_diagonal>(-1);
    }
}

constexpr rocblas_side char2rocblas_side(char value)
{
    switch(value)
    {
    case 'L':
    case 'l':
        return rocblas_side_left;
    case 'R':
    case 'r':
        return rocblas_side_right;
    default:
        return static_cast<rocblas_side>(-1);
    }
}

inline rocblas_initialization string2rocblas_initialization(const std::string& value)
{
    // clang-format off
    return
        value == "rand_int"   ? rocblas_initialization_random_int :
        value == "rand_narrow" ? rocblas_initialization_random_narrow:
        value == "rand_broad" ? rocblas_initialization_random_broad:
        value == "rand_full" ? rocblas_initialization_random_full:
        value == "trig_float" ? rocblas_initialization_trig_float :
        value == "hpl"        ? rocblas_initialization_hpl        :
        value == "const"        ? rocblas_initialization_const        :
        value == "file"        ? rocblas_initialization_file      :
        
        static_cast<rocblas_initialization>(-1);
    // clang-format on
}

inline rocblas_datatype string2rocblas_datatype(const std::string& value)
{
    // clang-format off
    return
        value == "f16_r" || value == "h" ? rocblas_datatype_f16_r :
        value == "f32_r" || value == "s" ? rocblas_datatype_f32_r :
        value == "f64_r" || value == "d" ? rocblas_datatype_f64_r :
        value == "bf16_r"                ? rocblas_datatype_bf16_r :
        value == "f16_c"                 ? rocblas_datatype_f16_c :
        value == "f32_c" || value == "c" ? rocblas_datatype_f32_c :
        value == "f64_c" || value == "z" ? rocblas_datatype_f64_c :
        value == "bf16_c"                ? rocblas_datatype_bf16_c :
        value == "i8_r"                  ? rocblas_datatype_i8_r  :
        value == "i32_r"                 ? rocblas_datatype_i32_r :
        value == "i8_c"                  ? rocblas_datatype_i8_c  :
        value == "i32_c"                 ? rocblas_datatype_i32_c :
        value == "u8_r"                  ? rocblas_datatype_u8_r  :
        value == "u32_r"                 ? rocblas_datatype_u32_r :
        value == "u8_c"                  ? rocblas_datatype_u8_c  :
        value == "u32_c"                 ? rocblas_datatype_u32_c :
        static_cast<rocblas_datatype>(-1);
    // clang-format on
}

class rocblas_local_handle
{
    rocblas_handle handle;

public:
    rocblas_local_handle()
    {
        rocblas_create_handle(&handle);
    }
    ~rocblas_local_handle()
    {
        rocblas_destroy_handle(handle);
    }

    // Allow rocblas_local_handle to be used anywhere rocblas_handle is expected
    operator rocblas_handle&()
    {
        return handle;
    }
    operator const rocblas_handle&() const
    {
        return handle;
    }
};

struct Arguments
{
    rocblas_int M;
    rocblas_int N;
    rocblas_int K;

    rocblas_int lda;
    rocblas_int ldb;
    rocblas_int ldc;
    rocblas_int ldd;

    rocblas_datatype a_type;
    rocblas_datatype b_type;
    rocblas_datatype c_type;
    rocblas_datatype d_type;
    rocblas_datatype compute_type;

    rocblas_int incx;
    rocblas_int incy;
    rocblas_int incd;
    rocblas_int incb;

    double alpha;
    double alphai;
    double beta;
    double betai;

    char transA;
    char transB;
    char side;
    char uplo;
    char diag;

    rocblas_int batch_count;

    rocblas_stride stride_a; //  stride_a > transA == 'N' ? lda * K : lda * M
    rocblas_stride stride_b; //  stride_b > transB == 'N' ? ldb * N : ldb * K
    rocblas_stride stride_c; //  stride_c > ldc * N
    rocblas_stride stride_d; //  stride_d > ldd * N

    double initVal;

    rocblas_int norm_check;
    rocblas_int unit_check;
    rocblas_int timing;
    rocblas_int iters;
    rocblas_int reinit_c;
    rocblas_int flush_gpu_cache;
    rocblas_int c_equals_d;
    rocblas_int time_each_iter;
    rocblas_int tensile_timing;

    uint32_t algo;
    int32_t  solution_index;
    uint32_t flags;

    char function[64];
    char name[64];
    char category[32];

    rocblas_initialization initialization;

    // Validate input format.
    // rocblas_gentest.py is expected to conform to this format.
    // rocblas_gentest.py uses rocblas_common.yaml to generate this format.
    static void validate(std::istream& ifs)
    {
        auto error = [](auto name) {
            std::cerr << "Arguments field " << name << " does not match format.\n\n"
                      << "Fatal error: Binary test data does match input format.\n"
                         "Ensure that rocblas_arguments.hpp and rocblas_common.yaml\n"
                         "define exactly the same Arguments, that rocblas_gentest.py\n"
                         "generates the data correctly, and that endianness is the same.\n";
            abort();
        };

        char      header[8] {}, trailer[8] {};
        Arguments arg {};
        ifs.read(header, sizeof(header));
        ifs >> arg;
        ifs.read(trailer, sizeof(trailer));

        if(strcmp(header, "rocBLAS"))
            error("header");
        else if(strcmp(trailer, "ROCblas"))
            error("trailer");

        auto check_func = [&, sig = (unsigned char)0](const auto& elem, auto name) mutable {
            static_assert(sizeof(elem) <= 255,
                          "One of the fields of Arguments is too large (> 255 bytes)");
            for(unsigned char i = 0; i < sizeof(elem); ++i)
                if(reinterpret_cast<const unsigned char*>(&elem)[i] ^ sig ^ i)
                    error(name);
            sig += 89;
        };

#define ROCBLAS_FORMAT_CHECK(x) check_func(arg.x, #x)

        // Order is important
        ROCBLAS_FORMAT_CHECK(M);
        ROCBLAS_FORMAT_CHECK(N);
        ROCBLAS_FORMAT_CHECK(K);
        ROCBLAS_FORMAT_CHECK(lda);
        ROCBLAS_FORMAT_CHECK(ldb);
        ROCBLAS_FORMAT_CHECK(ldc);
        ROCBLAS_FORMAT_CHECK(ldd);
        ROCBLAS_FORMAT_CHECK(a_type);
        ROCBLAS_FORMAT_CHECK(b_type);
        ROCBLAS_FORMAT_CHECK(c_type);
        ROCBLAS_FORMAT_CHECK(d_type);
        ROCBLAS_FORMAT_CHECK(initVal);
        ROCBLAS_FORMAT_CHECK(compute_type);
        ROCBLAS_FORMAT_CHECK(incx);
        ROCBLAS_FORMAT_CHECK(incy);
        ROCBLAS_FORMAT_CHECK(incd);
        ROCBLAS_FORMAT_CHECK(incb);
        ROCBLAS_FORMAT_CHECK(alpha);
        ROCBLAS_FORMAT_CHECK(alphai);
        ROCBLAS_FORMAT_CHECK(beta);
        ROCBLAS_FORMAT_CHECK(betai);
        ROCBLAS_FORMAT_CHECK(transA);
        ROCBLAS_FORMAT_CHECK(transB);
        ROCBLAS_FORMAT_CHECK(side);
        ROCBLAS_FORMAT_CHECK(uplo);
        ROCBLAS_FORMAT_CHECK(diag);
        ROCBLAS_FORMAT_CHECK(batch_count);
        ROCBLAS_FORMAT_CHECK(stride_a);
        ROCBLAS_FORMAT_CHECK(stride_b);
        ROCBLAS_FORMAT_CHECK(stride_c);
        ROCBLAS_FORMAT_CHECK(stride_d);
        ROCBLAS_FORMAT_CHECK(norm_check);
        ROCBLAS_FORMAT_CHECK(unit_check);
        ROCBLAS_FORMAT_CHECK(timing);
        ROCBLAS_FORMAT_CHECK(iters);
        ROCBLAS_FORMAT_CHECK(reinit_c);
        ROCBLAS_FORMAT_CHECK(flush_gpu_cache);
        ROCBLAS_FORMAT_CHECK(c_equals_d);
        ROCBLAS_FORMAT_CHECK(time_each_iter);
        ROCBLAS_FORMAT_CHECK(tensile_timing);
        ROCBLAS_FORMAT_CHECK(algo);
        ROCBLAS_FORMAT_CHECK(solution_index);
        ROCBLAS_FORMAT_CHECK(flags);
        ROCBLAS_FORMAT_CHECK(function);
        ROCBLAS_FORMAT_CHECK(name);
        ROCBLAS_FORMAT_CHECK(category);
        ROCBLAS_FORMAT_CHECK(initialization);
    }

    template <typename T>
    T get_alpha() const
    {
        return rocblas_isnan(alpha) || rocblas_isnan(alphai) ? T(0)
                                                             : convert_alpha_beta<T>(alpha, alphai);
    }

    template <typename T>
    T get_beta() const
    {
        return rocblas_isnan(beta) || rocblas_isnan(betai) ? T(0)
                                                           : convert_alpha_beta<T>(beta, betai);
    }

private:
    template <typename T, typename U, typename std::enable_if<!is_complex<T>, int>::type = 0>
    static T convert_alpha_beta(U r, U i)
    {
        return T(r);
    }

    template <typename T, typename U, typename std::enable_if<+is_complex<T>, int>::type = 0>
    static T convert_alpha_beta(U r, U i)
    {
        return T(r, i);
    }

    // Function to read Structures data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg)
    {
        str.read(reinterpret_cast<char*>(&arg), sizeof(arg));
        return str;
    }

    // print_value is for formatting different data types

    // Default output
    template <typename T>
    static void print_value(std::ostream& str, const T& x)
    {
        str << x;
    }

    // Floating-point output
    static void print_value(std::ostream& str, double x)
    {
        if(std::isnan(x))
            str << ".nan";
        else if(std::isinf(x))
            str << (x < 0 ? "-.inf" : ".inf");
        else
        {
            char s[32];
            snprintf(s, sizeof(s) - 2, "%.17g", x);

            // If no decimal point or exponent, append .0
            char* end = s + strcspn(s, ".eE");
            if(!*end)
                strcat(end, ".0");
            str << s;
        }
    }

    // Character output
    static void print_value(std::ostream& str, char c)
    {
        char s[] {c, 0};
        str << std::quoted(s, '\'');
    }

    // bool output
    static void print_value(std::ostream& str, bool b)
    {
        str << (b ? "true" : "false");
    }

    // string output
    static void print_value(std::ostream& str, const char* s)
    {
        str << std::quoted(s);
    }

    // Function to print Arguments out to stream in YAML format
    // Google Tests uses this automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg)
    {
        // delim starts as '{' opening brace and becomes ',' afterwards
        auto print = [&, delim = '{'](const char* name, auto x) mutable {
            str << delim << " " << name << ": ";
            print_value(str, x);
            delim = ',';
        };

        print("function", arg.function);
        print("a_type", rocblas_datatype2string(arg.a_type));
        print("b_type", rocblas_datatype2string(arg.b_type));
        print("c_type", rocblas_datatype2string(arg.c_type));
        print("d_type", rocblas_datatype2string(arg.d_type));
        print("compute_type", rocblas_datatype2string(arg.compute_type));
        print("transA", arg.transA);
        print("transB", arg.transB);
        print("M", arg.M);
        print("N", arg.N);
        print("K", arg.K);
        print("lda", arg.lda);
        print("ldb", arg.ldb);
        print("ldc", arg.ldc);
        print("ldd", arg.ldd);
        print("incx", arg.incx);
        print("incy", arg.incy);
        print("incd", arg.incd);
        print("incb", arg.incb);
        print("alpha", arg.alpha);
        print("alphai", arg.alphai);
        print("beta", arg.beta);
        print("betai", arg.betai);
        print("side", arg.side);
        print("uplo", arg.uplo);
        print("diag", arg.diag);
        print("batch_count", arg.batch_count);
        print("stride_a", arg.stride_a);
        print("stride_b", arg.stride_b);
        print("stride_c", arg.stride_c);
        print("stride_d", arg.stride_d);
        print("initVal", arg.initVal);
        print("algo", arg.algo);
        print("solution_index", arg.solution_index);
        print("flags", arg.flags);
        print("name", arg.name);
        print("category", arg.category);
        print("norm_check", arg.norm_check);
        print("unit_check", arg.unit_check);
        print("timing", arg.timing);
        print("iters", arg.iters);
        print("reinit_c", arg.reinit_c);
        print("flush_gpu_cache", arg.flush_gpu_cache);
        print("c_equals_d", arg.c_equals_d);
        print("time_each_iter", arg.time_each_iter);
        print("tensile_timing", arg.tensile_timing);
        print("initialization", arg.initialization);

        return str << " }\n";
    }
};

static_assert(std::is_standard_layout<Arguments> {},
              "Arguments is not a standard layout type, and thus is incompatible with C.");

static_assert(std::is_trivially_copyable<Arguments> {},
              "Arguments is not a trivially copyable type, and thus is incompatible with C.");

std::string function;
std::string precision;
std::string a_type;
std::string b_type;
std::string c_type;
std::string d_type;
std::string compute_type;
std::string initialization;
std::string a_file, b_file, c_file, o_file;
rocblas_int storeInitData;
rocblas_int storeOutputData;
rocblas_int device_id;

void readArgs(int argc, char* argv[], Arguments& arg)
{
    boost::program_options::options_description desc("rocblas-bench command line options");
    desc.add_options()
        // clang-format off
    ("sizem,m",
        po::value<rocblas_int>(&arg.M)->default_value(128),
        "Specific matrix size: sizem is only applicable to BLAS-2 & BLAS-3: the number of "
        "rows or columns in matrix.")

    ("sizen,n",
        po::value<rocblas_int>(&arg.N)->default_value(128),
        "Specific matrix/vector size: BLAS-1: the length of the vector. BLAS-2 & "
        "BLAS-3: the number of rows or columns in matrix")

    ("sizek,k",
        po::value<rocblas_int>(&arg.K)->default_value(128),
        "Specific matrix size:sizek is only applicable to BLAS-3: the number of columns in "
        "A and rows in B.")

    ("lda",
        po::value<rocblas_int>(&arg.lda)->default_value(128),
        "On entry, LDA specifies the first dimension of A as declared"
           "in the calling (sub) program. When  TRANSA = 'N' or 'n' then"
           "LDA must be at least  max( 1, m ), otherwise  LDA must be at"
           "least  max( 1, k )")

    ("ldb",
        po::value<rocblas_int>(&arg.ldb)->default_value(128),
        "On entry, LDB specifies the first dimension of B as declared"
           "in the calling (sub) program. When  TRANSB = 'N' or 'n' then"
           "LDB must be at least  max( 1, k ), otherwise  LDB must be at"
           "least  max( 1, n ).")

    ("ldc",
        po::value<rocblas_int>(&arg.ldc)->default_value(128),
        "On entry, LDC specifies the first dimension of C as declared"
           "in  the  calling  (sub)  program.   LDC  must  be  at  least"
           "max( 1, m ).")

    ("ldd",
        po::value<rocblas_int>(&arg.ldd)->default_value(128),
        "On entry, LDD specifies the first dimension of D as desired"
           "in  the  calling  (sub)  program.   LDD  must  be  at  least"
           "max( 1, m ).")

    ("stride_a",
        po::value<rocblas_stride>(&arg.stride_a)->default_value(128*128),
        "Specific stride of strided_batched matrix A, is only applicable to strided batched"
        "BLAS-2 and BLAS-3: second dimension * leading dimension.")

    ("stride_b",
        po::value<rocblas_stride>(&arg.stride_b)->default_value(128*128),
        "Specific stride of strided_batched matrix B, is only applicable to strided batched"
        "BLAS-2 and BLAS-3: second dimension * leading dimension.")

    ("stride_c",
        po::value<rocblas_stride>(&arg.stride_c)->default_value(128*128),
        "Specific stride of strided_batched matrix C, is only applicable to strided batched"
        "BLAS-2 and BLAS-3: second dimension * leading dimension.")

    ("stride_d",
        po::value<rocblas_stride>(&arg.stride_d)->default_value(128*128),
        "Specific stride of strided_batched matrix D, is only applicable to strided batched"
        "BLAS_EX: second dimension * leading dimension.")

    ("alpha",
        po::value<double>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

    ("beta",
        po::value<double>(&arg.beta)->default_value(0.0), "specifies the scalar beta")
        
    ("initVal",
    po::value<double>(&arg.initVal)->default_value(1.0), "specifies the const value to be used with const initialization")

    ("function,f",
        po::value<std::string>(&function)->default_value("gemm"),
        "GEMM function to test. (gemm, gemm_strided_batched and gemm_ex")

    ("precision,r",
        po::value<std::string>(&precision)->default_value("f32_r"), "Precision. "
        "Options: s,d,f16_r,bf16_r,f32_r,f64_r")

    ("a_type",
        po::value<std::string>(&a_type), "Precision of matrix A. "
        "Options: s,d,f32_r,f64_r")

    ("b_type",
        po::value<std::string>(&b_type), "Precision of matrix B. "
        "Options: s,d,f32_r,f64_r")

    ("c_type",
        po::value<std::string>(&c_type), "Precision of matrix C. "
        "Options: s,d,f32_r,f64_r")

    ("d_type",
        po::value<std::string>(&d_type), "Precision of matrix D. "
        "Options: s,d,f32_r,f64_r")

    ("compute_type",
        po::value<std::string>(&compute_type), "Precision of computation. "
        "Options: s,d,b16_r,f32_r,f64_r")

    ("initialization",
        po::value<std::string>(&initialization)->default_value("rand_int"),
        "Intialize with random numbers, trig functions sin and cos, hpl-like input, or by loading data from a bin file"
        "Options: rand_int, rand_narrow, rand_broad, rand_full, trig_float, hpl, const, file")

    ("storeInitData,s",
        po::value<rocblas_int>(&storeInitData)->default_value(0),
        "Dump initialization data in to bin files? Note: Storing is not done when loading from bin files. "
        "Please specify file names using --x_file flags 0 = No, 1 = Yes (default: No)")

    ("storeOutputData,o",
        po::value<rocblas_int>(&storeOutputData)->default_value(0),
        "Dump results matrix in to bin files?"
        "Please specify file names using --o_file flag 0 = No, 1 = Yes (default: No) "
        "Note that multiple iterations will change results unless reinit_c flag is specified")

    ("a_file",
        po::value<std::string>(&a_file)->default_value(""), "Bin file storing matrix A. "
        "Options: text.bin")

    ("b_file",
        po::value<std::string>(&b_file)->default_value(""), "Bin file storing matrix B. "
        "Options: text.bin")

    ("c_file",
        po::value<std::string>(&c_file)->default_value(""), "Bin file storing matrix C. "
        "Options: text.bin")

    ("o_file",
        po::value<std::string>(&o_file)->default_value(""), "Bin file storing result matrix. "
        "Options: text.bin")

    ("transposeA",
        po::value<char>(&arg.transA)->default_value('N'),
        "N = no transpose, T = transpose, C = conjugate transpose")

    ("transposeB",
        po::value<char>(&arg.transB)->default_value('N'),
        "N = no transpose, T = transpose, C = conjugate transpose")

    ("batch",
        po::value<rocblas_int>(&arg.batch_count)->default_value(1),
        "Number of matrices. Only applicable to batched routines") // xtrsm xtrsm_ex xtrmm xgemm

    ("verify,v",
        po::value<rocblas_int>(&arg.norm_check)->default_value(0),
        "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

    ("unit_check,u",
        po::value<rocblas_int>(&arg.unit_check)->default_value(0),
        "Unit Check? 0 = No, 1 = Yes (default: No)")

    ("iters,i",
        po::value<rocblas_int>(&arg.iters)->default_value(10),
        "Iterations to run inside timing loop")
    
    ("reinit_c",
        po::value<rocblas_int>(&arg.reinit_c)->default_value(0),
        "Reinitialize C between iterations? 0 = No, 1 = Yes (default: No)")

    ("flush_gpu_cache",
        po::value<rocblas_int>(&arg.flush_gpu_cache)->default_value(0),
        "Flush GPU L2 cache between iterations? 0 = No, 1 = Yes (default: No)")

    ("c_equals_d",
        po::value<rocblas_int>(&arg.c_equals_d)->default_value(0),
        "is C equal to D? 0 = No, 1 = Yes (default: No)")

    ("time_each_iter",
        po::value<rocblas_int>(&arg.time_each_iter)->default_value(0),
        "Explicitly time each iteration? 0 = No, 1 = Yes (default: No)")

    ("tensile_timing",
        po::value<rocblas_int>(&arg.tensile_timing)->default_value(0),
        "Get kernel timing from Tensile? This sends hipEvents directly to the kernel call,"
        " eliminating overhead that may be seen for smaller launches. "
        "Will use this timing to calculate performance when enabled.\n"
         "Options: 0 = No, 1 = Yes (default: No)")

    ("algo",
        po::value<uint32_t>(&arg.algo)->default_value(0),
        "extended precision gemm algorithm")

    ("solution_index",
        po::value<int32_t>(&arg.solution_index)->default_value(0),
        "extended precision gemm solution index")

    ("flags",
        po::value<uint32_t>(&arg.flags)->default_value(10),
        "extended precision gemm flags")

    ("device",
        po::value<rocblas_int>(&device_id)->default_value(0),
        "Set default device to be used for subsequent program runs (default: 0)")

    ("help,h", "produces this help message");

    // clang-format on

    po::variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(1);
    }

    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
    auto prec = string2rocblas_datatype(precision);
    if(prec == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --precision " + precision);

    if(a_type == "")
        a_type = precision;
    arg.a_type = string2rocblas_datatype(a_type);
    if(arg.a_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --a_type " + a_type);

    if(b_type == "")
        b_type = precision;
    arg.b_type = string2rocblas_datatype(b_type);
    if(arg.b_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --b_type " + b_type);

    if(c_type == "")
        c_type = precision;
    arg.c_type = string2rocblas_datatype(c_type);
    if(arg.c_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --c_type " + c_type);

    if(d_type == "")
        d_type = precision;
    arg.d_type = string2rocblas_datatype(d_type);
    if(arg.d_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --d_type " + d_type);

    if(compute_type == "")
        compute_type = precision;
    arg.compute_type = string2rocblas_datatype(compute_type);
    if(arg.compute_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --compute_type " + compute_type);

    arg.initialization = string2rocblas_initialization(initialization);
    if(arg.initialization == static_cast<rocblas_initialization>(-1))
        throw std::invalid_argument("Invalid value for --initialization " + initialization);

    if(arg.M < 0)
        throw std::invalid_argument("Invalid value for -m " + std::to_string(arg.M));
    if(arg.N < 0)
        throw std::invalid_argument("Invalid value for -n " + std::to_string(arg.N));
    if(arg.K < 0)
        throw std::invalid_argument("Invalid value for -k " + std::to_string(arg.K));

    if(arg.initialization
       == rocblas_initialization_file) //check for files if initialization is file
    {
        if(!std::ifstream(a_file))
            throw std::invalid_argument("Invalid value for --a_file " + a_file);
        if(!std::ifstream(b_file))
            throw std::invalid_argument("Invalid value for --b_file " + b_file);
        if(!std::ifstream(c_file))
            throw std::invalid_argument("Invalid value for --c_file " + c_file);
    }
    else if(arg.initialization == rocblas_initialization_const && std::isnan(arg.initVal))
    {
        throw std::invalid_argument("Invalid value for --initVal " + std::to_string(arg.initVal));
    }

    if(storeInitData)
    {
        if(arg.initialization == rocblas_initialization_file)
        {
            storeInitData = 0; //Do not store if loading from file
        }
        else
        {
            if(a_file.empty())
                throw std::invalid_argument("Invalid value for --a_file " + a_file);
            if(b_file.empty())
                throw std::invalid_argument("Invalid value for --b_file " + b_file);
            if(c_file.empty())
                throw std::invalid_argument("Invalid value for --c_file " + c_file);
        }
    }

    if(storeOutputData)
    {
        if(o_file.empty())
            throw std::invalid_argument("Invalid value for --o_file " + o_file);
    }

    // Device Query
    rocblas_int device_count = query_device_property();

    if(device_count <= device_id)
        throw std::invalid_argument("Invalid Device ID");
    set_device(device_id);
}

template <typename Ti, typename To = Ti>
void loadFromBin(rocblas_operation transA,
                 rocblas_operation transB,
                 rocblas_int       M,
                 rocblas_int       N,
                 rocblas_int       K,
                 host_vector<Ti>&   hA,
                 rocblas_int       lda,
                 std::string       ADataFile,
                 host_vector<Ti>&   hB,
                 rocblas_int       ldb,
                 std::string       BDataFile,
                 host_vector<To>&   hC,
                 rocblas_int       ldc,
                 std::string       CDataFile,
                 rocblas_int       batch_count)
{
    {
        size_t sz = lda * (transA == rocblas_operation_none ? K : M) * sizeof(Ti) * batch_count;
        std::ifstream FILE(ADataFile, std::ios::in | std::ofstream::binary);
        FILE.seekg(0, FILE.end);
        int fileLength = FILE.tellg();
        FILE.seekg(0, FILE.beg);

        if(sz > fileLength)
        {
            std::cout << "Binary file bytes " << fileLength << " Gemm required bytes " << sz
                      << std::endl;
            std::cout << "Not enough elements in A data file...exiting" << std::endl;
            exit(1);
        }
        FILE.read(reinterpret_cast<char*>(&hA[0]), sz);
    }

    {
        size_t sz = ldb * (transB == rocblas_operation_none ? N : K) * sizeof(Ti) * batch_count;
        std::ifstream FILE(BDataFile, std::ios::in | std::ofstream::binary);
        FILE.seekg(0, FILE.end);
        int fileLength = FILE.tellg();
        FILE.seekg(0, FILE.beg);

        if(sz > fileLength)
        {
            std::cout << "Binary file bytes " << fileLength << " Gemm required bytes " << sz
                      << std::endl;
            std::cout << "Not enough elements in B data file...exiting" << std::endl;
            exit(1);
        }
        FILE.read(reinterpret_cast<char*>(&hB[0]), sz);
    }

    {
        size_t        sz = ldc * N * sizeof(To) * batch_count;
        std::ifstream FILE(CDataFile, std::ios::in | std::ofstream::binary);
        FILE.seekg(0, FILE.end);
        int fileLength = FILE.tellg();
        FILE.seekg(0, FILE.beg);

        if(sz > fileLength)
        {
            std::cout << "Binary file bytes " << fileLength << " Gemm required bytes " << sz
                      << std::endl;
            std::cout << "Not enough elements in C data file...exiting" << std::endl;
            exit(1);
        }
        FILE.read(reinterpret_cast<char*>(&hC[0]), sz);
    }
}

template <typename Ti, typename To>
void storeInitToBin(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int       M,
                rocblas_int       N,
                rocblas_int       K,
                host_vector<Ti>&   hA,
                rocblas_int       lda,
                std::string       ADataFile,
                host_vector<Ti>&   hB,
                rocblas_int       ldb,
                std::string       BDataFile,
                host_vector<To>&   hC,
                rocblas_int       ldc,
                std::string       CDataFile,
                rocblas_int       batch_count)
{
    {
        size_t sz = lda * (transA == rocblas_operation_none ? K : M) * sizeof(Ti) * batch_count;
        std::ofstream FILE(ADataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&hA[0]), sz);
    }

    {
        size_t sz = ldb * (transB == rocblas_operation_none ? N : K) * sizeof(Ti) * batch_count;
        std::ofstream FILE(BDataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&hB[0]), sz);
    }

    {
        size_t        sz = ldc * N * sizeof(To) * batch_count;
        std::ofstream FILE(CDataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&hC[0]), sz);
    }
}

template <typename To>
void storeOutputToBin(rocblas_int       N,
                host_vector<To>&   hO,
                rocblas_int       ldo,
                std::string       ODataFile,
                rocblas_int       batch_count)
{
    {
        size_t        sz = ldo * N * sizeof(To) * batch_count;
        std::ofstream FILE(ODataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&hO[0]), sz);
    }
}

static std::mt19937 rng;

template <typename>
struct FP_PARAM;

template <>
struct FP_PARAM<double>
{
    using UINT_T              = uint64_t;
    static constexpr int NSIG = 52;
    static constexpr int NEXP = 11;
};

template <>
struct FP_PARAM<float>
{
    using UINT_T              = uint32_t;
    static constexpr int NSIG = 23;
    static constexpr int NEXP = 8;
};

template <>
struct FP_PARAM<rocblas_bfloat16>
{
    using UINT_T              = uint16_t;
    static constexpr int NSIG = 7;
    static constexpr int NEXP = 8;
};

template <>
struct FP_PARAM<rocblas_half>
{
    using UINT_T              = uint16_t;
    static constexpr int NSIG = 10;
    static constexpr int NEXP = 5;
};

template <typename T>
struct rocm_random_common : FP_PARAM<T>
{
    using typename FP_PARAM<T>::UINT_T;
    using FP_PARAM<T>::NSIG;
    using FP_PARAM<T>::NEXP;
    using random_fp_int_dist = std::uniform_int_distribution<UINT_T>;

    static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
    static constexpr UINT_T expmask = (((UINT_T)1 << NEXP) - 1) << NSIG;
    static constexpr UINT_T expbias = ((UINT_T)1 << (NEXP - 1)) - 1;
    static T                signsig_exp(UINT_T signsig, UINT_T exp)
    {
        union
        {
            UINT_T u;
            T      fp;
        };
        u = signsig & ~expmask | (exp + expbias << NSIG) & expmask;
        return fp;
    }
};

template <typename T, int LOW_EXP, int HIGH_EXP>
struct rocm_random : rocm_random_common<T>
{
    using typename rocm_random_common<T>::random_fp_int_dist;
    __attribute__((flatten)) T operator()()
    {
        int exp = std::uniform_int_distribution<int> {}(rng);
        exp     = exp % (HIGH_EXP - LOW_EXP + 1) + LOW_EXP;
        return this->signsig_exp(random_fp_int_dist {}(rng), exp);
    }
};

// These values should be squareable and not overflow/underflow
template <typename T>
struct rocm_random_squareable;

template <>
struct rocm_random_squareable<double> : rocm_random<double, -189, 200>
{
};

template <>
struct rocm_random_squareable<float> : rocm_random<float, -100, 100>
{
};

template <>
struct rocm_random_squareable<rocblas_bfloat16> : rocm_random<rocblas_bfloat16, -100, 100>
{
};

template <>
struct rocm_random_squareable<rocblas_half> : rocm_random<rocblas_half, -100, 100>
{
};

template <typename T>
using rocm_random_full_range = rocm_random<T,
                                           -((int)((uint64_t)1 << (FP_PARAM<T>::NEXP - 1)) - 1),
                                           (uint32_t)((uint64_t)1 << (FP_PARAM<T>::NEXP - 1)) - 1>;

// These values, when scaled, should be addable without loss
// Basically for these values x, a(1+x) should be different than a.
template <typename T>
using rocm_random_addable = rocm_random<T, 1, FP_PARAM<T>::NSIG>;

template <typename T>
struct rocm_random_narrow_range;

template <>
struct rocm_random_narrow_range<double> : rocm_random<double, -189, 0>
{
};

template <>
struct rocm_random_narrow_range<float> : rocm_random<float, -100, 0>
{
};

template <>
struct rocm_random_narrow_range<rocblas_bfloat16> : rocm_random<rocblas_bfloat16, -100, 0>
{
};

template <>
struct rocm_random_narrow_range<rocblas_half> : rocm_random<rocblas_half, -100, 0>
{
};

// Values from 1-2
template <typename T>
using rocm_random_1_2 = rocm_random<T, 0, 1>;

template <template <typename> class RAND, typename T>
static void init_matrix( std::vector<T>& mat, size_t m, size_t n, size_t ld, rocblas_stride stride, size_t batch)
{
    for(size_t i = 0; i < batch; ++i)
        for(size_t j = 0; j < n; ++j)
            for(size_t k = 0; k < m; ++k)
                mat[i * stride + j * ld + k] = RAND<T> {}();
}

template <typename T>
static void
    init_constant_matrix( std::vector<T>& mat, size_t m, size_t n, size_t ld, rocblas_stride stride, size_t batch, T val)
{
    for(size_t i = 0; i < batch; i++)
        for(size_t j = 0; j < n; ++j)
            for(size_t k = 0; k < m; ++k)
                mat[i * stride + j * ld + k] = val;
}

// Absolute value
template <typename T, typename std::enable_if<!is_complex<T>, int>::type = 0>
__device__ __host__ inline T rocblas_abs(T x)
{
    return x < 0 ? -x : x;
}

// For complex, we have defined a __device__ __host__ compatible std::abs
template <typename T, typename std::enable_if<is_complex<T>, int>::type = 0>
__device__ __host__ inline auto rocblas_abs(T x)
{
    return std::abs(x);
}

// rocblas_half
__device__ __host__ inline rocblas_half rocblas_abs(rocblas_half x)
{
    union
    {
        rocblas_half x;
        uint16_t     data;
    } t = {x};
    t.data &= 0x7fff;
    return t.x;
}

// rocblas_bfloat16 is handled specially
__device__ __host__ inline rocblas_bfloat16 rocblas_abs(rocblas_bfloat16 x)
{
    x.data &= 0x7fff;
    return x;
}

// Output rocblas_half value
inline std::ostream& operator<<(std::ostream& os, rocblas_half x)
{
    return os << float(x);
}

template <typename T>
void normalizeInputs(rocblas_operation transa,
                     rocblas_operation transb,
                     size_t            m,
                     size_t            n,
                     size_t            k,
                      std::vector<T>& a,
                     size_t lda,
                     rocblas_stride stride_a,
                      std::vector<T>& b,
                     size_t ldb,
                     rocblas_stride stride_b,
                      std::vector<T>& c,
                     size_t ldc,
                     rocblas_stride stride_c,
                     size_t batch)
{
    // We divide each element of B by the maximum corresponding element of A such that elem(A * B) <
    // 2 ** NSIG
    if(transa == rocblas_operation_none)
    {
        for(size_t i = 0; i < batch; i++)
        {
            for(size_t j = 0; j < k; ++j)
            {
                T scal = T(0);
                for(size_t k = 0; k < m; ++k)
                {
                    T val = T(rocblas_abs(a[i * stride_a + j * lda + k]));
                    if(val > scal)
                        scal = val;
                }

                if(!scal)
                    abort();

                scal = T(1) / scal;
                if(transb == rocblas_operation_none)
                    for(size_t k = 0; k < n; ++k)
                        b[i * stride_b + j * ldb + k] *= scal;
                else
                    for(size_t k = 0; k < n; ++k)
                        b[i * stride_b + k * ldb + j] *= scal;
            }
        }
    }
    else
    {
        for(size_t i = 0; i < batch; i++)
        {
            for(size_t j = 0; j < k; ++j)
            {
                T scal = T(0);
                for(size_t k = 0; k < m; ++k)
                {
                    T val = T(rocblas_abs(a[i * stride_a + k * lda + j]));
                    if(val > scal)
                        scal = val;
                }

                if(!scal)
                    abort();

                scal = T(1) / scal;
                if(transb == rocblas_operation_none)
                    for(size_t k = 0; k < n; ++k)
                        b[i * stride_b + j * ldb + k] *= scal;
                else
                    for(size_t k = 0; k < n; ++k)
                        b[i * stride_b + k * ldb + j] *= scal;
            }
        }
    }
}

template <typename T, typename U = T>
static void init_broad_range_random_gemm(rocblas_operation transa,
                                         rocblas_operation transb,
                                         size_t            m,
                                         size_t            n,
                                         size_t            k,
                                         std::vector<T>& a,
                                         size_t lda,
                                         rocblas_stride stride_a,
                                         std::vector<T>& b,
                                         size_t ldb,
                                         rocblas_stride stride_b,
                                         std::vector<U>& c,
                                         size_t ldc,
                                         rocblas_stride stride_c,
                                         size_t batch = 1)
{
    init_matrix<rocm_random_squareable>(a, transa == rocblas_operation_none ? m : k, transa == rocblas_operation_none ? k : m, lda, stride_a, batch);
    init_matrix<rocm_random_addable>(b, transb == rocblas_operation_none ? k : n, transb == rocblas_operation_none ? n : k, ldb, stride_b, batch);
    init_matrix<rocm_random_addable>(c, m, n, ldc, stride_c, batch);

    normalizeInputs<T>(
        transa, transb, m, n, k, a, lda, stride_a, b, ldb, stride_b, c, ldc, stride_c, batch);
}

template <typename T, typename U = T>
static void init_narrow_range_random_gemm(rocblas_operation transa,
                                          rocblas_operation transb,
                                          size_t            m,
                                          size_t            n,
                                          size_t            k,
                                          std::vector<T>& a,
                                          size_t lda,
                                          rocblas_stride stride_a,
                                          std::vector<T>& b,
                                          size_t ldb,
                                          rocblas_stride stride_b,
                                          std::vector<U>& c,
                                          size_t ldc,
                                          rocblas_stride stride_c,
                                          size_t batch = 1)
{
    init_matrix<rocm_random_narrow_range>(a, transa == rocblas_operation_none ? m : k, transa == rocblas_operation_none ? k : m, lda, stride_a, batch);
    init_matrix<rocm_random_narrow_range>(b, transb == rocblas_operation_none ? k : n, transb == rocblas_operation_none ? n : k, ldb, stride_b, batch);
    init_matrix<rocm_random_narrow_range>(c, m, n, ldc, stride_c, batch);
}

//can cause nans from overflow
template <typename T, typename U = T>
static void init_full_range_random_gemm(rocblas_operation transa,
                                        rocblas_operation transb,
                                        size_t            m,
                                        size_t            n,
                                        size_t            k,
                                        std::vector<T>& a,
                                        size_t lda,
                                        rocblas_stride stride_a,
                                        std::vector<T>& b,
                                        size_t ldb,
                                        rocblas_stride stride_b,
                                        std::vector<U>& c,
                                        size_t ldc,
                                        rocblas_stride stride_c,
                                        size_t batch = 1)
{
    init_matrix<rocm_random_full_range>(a, transa == rocblas_operation_none ? m : k, transa == rocblas_operation_none ? k : m, lda, stride_a, batch);
    init_matrix<rocm_random_full_range>(b, transb == rocblas_operation_none ? k : n, transb == rocblas_operation_none ? n : k, ldb, stride_b, batch);
    init_matrix<rocm_random_full_range>(c, m, n, ldc, stride_c, batch);
}

template <typename T, typename U = T>
static void init_constant_gemm(rocblas_operation transa,
                               rocblas_operation transb,
                               size_t            m,
                               size_t            n,
                               size_t            k,
                               std::vector<T>& a,
                               size_t lda,
                               rocblas_stride stride_a,
                               std::vector<T>& b,
                               size_t ldb,
                               rocblas_stride stride_b,
                               std::vector<U>& c,
                               size_t ldc,
                               rocblas_stride stride_c,
                               T      val,
                               size_t batch = 1)
{
    init_constant_matrix<T>(a, transa == rocblas_operation_none ? m : k, transa == rocblas_operation_none ? k : m, lda, stride_a, batch, val);
    init_constant_matrix<T>(b, transb == rocblas_operation_none ? k : n, transb == rocblas_operation_none ? n : k, ldb, stride_b, batch, val);
    init_constant_matrix<U>(c, m, n, ldc, stride_c, batch, U(val));
}

#endif /* _UTILITY_ */
