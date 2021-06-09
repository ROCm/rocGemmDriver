/* ************************************************************************
 * Copyright (c) <2021> Advanced Micro Devices, Inc.
 *  
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *  
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *  
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * ************************************************************************ */
#include "flame_interface.hpp"
#include <rocblas.h>

#define UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, UNIT_ASSERT_EQ)
#define ASSERT_HALF_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))

#define ASSERT_BFLOAT16_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))

#define ASSERT_FLOAT_COMPLEX_EQ(a, b)                  \
    do                                                 \
    {                                                  \
        auto ta = (a), tb = (b);                       \
        ASSERT_FLOAT_EQ(std::real(ta), std::real(tb)); \
        ASSERT_FLOAT_EQ(std::imag(ta), std::imag(tb)); \
    } while(0)

#define ASSERT_DOUBLE_COMPLEX_EQ(a, b)                  \
    do                                                  \
    {                                                   \
        auto ta = (a), tb = (b);                        \
        ASSERT_DOUBLE_EQ(std::real(ta), std::real(tb)); \
        ASSERT_DOUBLE_EQ(std::imag(ta), std::imag(tb)); \
    } while(0)

template <typename T>
void unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU);

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_bfloat16* hCPU, rocblas_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_half* hCPU, rocblas_half* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
inline void
    unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, float* hCPU, float* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
inline void
    unit_check_general(rocblas_int M, rocblas_int N, rocblas_int lda, double* hCPU, double* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int            M,
                               rocblas_int            N,
                               rocblas_int            lda,
                               rocblas_float_complex* hCPU,
                               rocblas_float_complex* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int             M,
                               rocblas_int             N,
                               rocblas_int             lda,
                               rocblas_double_complex* hCPU,
                               rocblas_double_complex* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
inline void unit_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, rocblas_int* hCPU, rocblas_int* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_EQ);
}

template <typename T>
void unit_check_general(rocblas_int    M,
                        rocblas_int    N,
                        rocblas_int    batch_count,
                        rocblas_int    lda,
                        rocblas_stride strideA,
                        T*             hCPU,
                        T*             hGPU);

template <>
inline void unit_check_general(rocblas_int       M,
                               rocblas_int       N,
                               rocblas_int       batch_count,
                               rocblas_int       lda,
                               rocblas_stride    strideA,
                               rocblas_bfloat16* hCPU,
                               rocblas_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
inline void unit_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               rocblas_half*  hCPU,
                               rocblas_half*  hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               float*         hCPU,
                               float*         hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               double*        hCPU,
                               double*        hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(rocblas_int            M,
                               rocblas_int            N,
                               rocblas_int            batch_count,
                               rocblas_int            lda,
                               rocblas_stride         strideA,
                               rocblas_float_complex* hCPU,
                               rocblas_float_complex* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int             M,
                               rocblas_int             N,
                               rocblas_int             batch_count,
                               rocblas_int             lda,
                               rocblas_stride          strideA,
                               rocblas_double_complex* hCPU,
                               rocblas_double_complex* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               rocblas_int*   hCPU,
                               rocblas_int*   hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_EQ);
}

/*! \brief  Template: norm check for hermitian/symmetric Matrix: half/float/double/complex */

template <typename T>
double norm_check_symmetric(
    char norm_type, char uplo, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU);

/* =====================================================================
        Norm check: norm(A-B)/norm(A), evaluate relative error
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Norm check
 */

/* ========================================Norm Check
 * ==================================================== */

// see check_norm.cpp for template speciliazation
// use auto as the return type is only allowed in c++14
// convert float/float to double

inline void xaxpy(int n, float alpha, float* x, int incx, float* y, int incy)
{
    return flame_saxpy(n, alpha, x, incx, y, incy);
}

inline void xaxpy(int n, double alpha, double* x, int incx, double* y, int incy)
{
    return flame_daxpy(n, alpha, x, incx, y, incy);
}

inline void xaxpy(
    int n, float alpha, rocblas_float_complex* x, int incx, rocblas_float_complex* y, int incy)
{
    return flame_caxpy(n, alpha, x, incx, y, incy);
}

inline void xaxpy(int                    n,
                  double                 alpha,
                  rocblas_double_complex* x,
                  int                    incx,
                  rocblas_double_complex* y,
                  int                    incy)
{
    return flame_zaxpy(n, alpha, x, incx, y, incy);
}

/*! \brief  Overloading: norm check for general Matrix: half/float/doubel/complex */
inline float xlange(char norm_type, int m, int n, float* A, int lda, float* work)
{
    return flame_slange(norm_type, m, n, A, lda, work);
}

inline double xlange(char norm_type, int m, int n, double* A, int lda, double* work)
{
    return flame_dlange(norm_type, m, n, A, lda, work);
}

inline float
    xlange(char norm_type, int m, int n, rocblas_float_complex* A, int lda, float* work)
{
    return flame_clange(norm_type, m, n, A, lda, work);
}

inline double
    xlange(char norm_type, int m, int n, rocblas_double_complex* A, int lda, double* work)
{
    return flame_zlange(norm_type, m, n, A, lda, work);
}

template <typename T>
void m_axpy(size_t N, T alpha, T *x, int incx, T *y, int incy) {
     for (size_t i=0; i < N; i++) {
             y[i*(incy)] = (alpha)*x[i*(incx)] + y[i*(incy)];
     }
}

/* ============== Norm Check for General Matrix ============= */
/*! \brief compare the norm error of two matrices hCPU & hGPU */

// Real
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
double norm_check_general(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU)
{
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    size_t size = N * (size_t)lda;

    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(rocblas_int i = 0; i < N; i++)
    {
        for(rocblas_int j = 0; j < M; j++)
        {
            size_t idx = j + i * (size_t)lda;
            hCPU_double[idx] = double(hCPU[idx]);
            hGPU_double[idx] = double(hGPU[idx]);
        }
    }

    double      work[1];
    rocblas_int incx  = 1;
    double      alpha = -1.0;

    double cpu_norm = xlange(norm_type, M, N, hCPU_double.data(), lda, work);
    m_axpy(size, alpha, hCPU_double.data(), incx, hGPU_double.data(), incx);
    double error = xlange(norm_type, M, N, hGPU_double.data(), lda, work) / cpu_norm;

    return error;
}

// Complex
template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
double norm_check_general(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    decltype(std::real(*hCPU)) work[1];
    rocblas_int                incx  = 1;
    T                          alpha = -1.0;
    size_t                     size  = N * (size_t)lda;

    double cpu_norm = xlange(norm_type, M, N, hCPU, lda, work);
    m_axpy(size, alpha, hCPU, incx, hGPU, incx);
    double error = xlange(norm_type, M, N, hGPU, lda, work) / cpu_norm;

    return error;
}

// For BF16 and half, we convert the results to double first
template <typename T,
          typename VEC,
          std::enable_if_t<std::is_same<T, rocblas_half>{} || std::is_same<T, rocblas_bfloat16>{},
                           int> = 0>
double norm_check_general(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, VEC hCPU, T* hGPU)
{
    size_t size  = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(rocblas_int i = 0; i < N; i++)
    {
        for(rocblas_int j = 0; j < M; j++)
        {
            size_t idx = j + i * (size_t)lda;
            hCPU_double[idx] = hCPU[idx];
            hGPU_double[idx] = hGPU[idx];
        }
    }

    return norm_check_general<double>(norm_type, M, N, lda, hCPU_double, hGPU_double);
}

/* ============== Norm Check for strided_batched case ============= */
template <typename T, template <typename> class VEC, typename T_hpa>
double norm_check_general(char           norm_type,
                          rocblas_int    M,
                          rocblas_int    N,
                          rocblas_int    lda,
                          rocblas_stride stride_a,
                          rocblas_int    batch_count,
                          VEC<T_hpa>    hCPU,
                          T*             hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(size_t i = 0; i < batch_count; i++)
    {
        auto index = i * stride_a;

        auto error = norm_check_general(norm_type, M, N, lda, (T_hpa*)hCPU + index, hGPU + index);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

/* ============== Norm Check for batched case ============= */

template <typename T, typename T_hpa>
double norm_check_general(char                      norm_type,
                          rocblas_int               M,
                          rocblas_int               N,
                          rocblas_int               lda,
                          rocblas_int    batch_count,
                          host_vector<T_hpa> hCPU[],
                          host_vector<T>     hGPU[])
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(rocblas_int i = 0; i < batch_count; i++)
    {
        auto index = i;

        auto error = norm_check_general<T>(norm_type, M, N, lda, hCPU[index], hGPU[index]);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

template <typename T>
double norm_check_general(char        norm_type,
                          rocblas_int M,
                          rocblas_int N,
                          rocblas_int lda,
                          rocblas_int    batch_count,
                          T*          hCPU[],
                          T*          hGPU[])
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(rocblas_int i = 0; i < batch_count; i++)
    {
        auto index = i;

        auto error = norm_check_general<T>(norm_type, M, N, lda, hCPU[index], hGPU[index]);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

template <class T>
static constexpr double sum_error_tolerance = 0.0;

template <>
static constexpr double sum_error_tolerance<rocblas_half> = 1 / 900.0;

#define NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, err, NEAR_ASSERT)

#define NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, err, NEAR_ASSERT)

#define ASSERT_NEAR(val1, val2, abs_error) \
    ASSERT_PRED_FORMAT3(::testing::internal::DoubleNearPredFormat, val1, val2, abs_error)

#define NEAR_ASSERT_HALF(a, b, err) ASSERT_NEAR(float(a), float(b), err)

template <typename T>
void near_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU, double abs_error);

template <>
inline void near_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, float* hCPU, float* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(
    rocblas_int M, rocblas_int N, rocblas_int lda, double* hCPU, double* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <typename T>
void near_check_general(rocblas_int M,
                        rocblas_int N,
                        rocblas_int batch_count,
                        rocblas_int lda,
                        rocblas_int strideA,
                        T*          hCPU,
                        T*          hGPU,
                        double      abs_error);

template <>
inline void near_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int batch_count,
                               rocblas_int lda,
                               rocblas_int strideA,
                               float*      hCPU,
                               float*      hGPU,
                               double      abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int M,
                               rocblas_int N,
                               rocblas_int batch_count,
                               rocblas_int lda,
                               rocblas_int strideA,
                               double*     hCPU,
                               double*     hGPU,
                               double      abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int   M,
                               rocblas_int   N,
                               rocblas_int   lda,
                               rocblas_half* hCPU,
                               rocblas_half* hGPU,
                               double        abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <typename T>
void near_check_general(rocblas_int    M,
                        rocblas_int    N,
                        rocblas_int    batch_count,
                        rocblas_int    lda,
                        rocblas_stride strideA,
                        T*             hCPU,
                        T*             hGPU,
                        double         abs_error);

template <>
inline void near_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               float*         hCPU,
                               float*         hGPU,
                               double         abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               double*        hCPU,
                               double*        hGPU,
                               double         abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int    M,
                               rocblas_int    N,
                               rocblas_int    batch_count,
                               rocblas_int    lda,
                               rocblas_stride strideA,
                               rocblas_half*  hCPU,
                               rocblas_half*  hGPU,
                               double         abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <typename T>
void near_check_general(rocblas_int    M,
                        rocblas_int    N,
                        rocblas_int    batch_count,
                        rocblas_int    lda,
                        host_vector<T> hCPU[],
                        host_vector<T> hGPU[],
                        double         abs_error);

template <>
inline void near_check_general(rocblas_int               M,
                               rocblas_int               N,
                               rocblas_int               batch_count,
                               rocblas_int               lda,
                               host_vector<rocblas_half> hCPU[],
                               host_vector<rocblas_half> hGPU[],
                               double                    abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general(rocblas_int        M,
                               rocblas_int        N,
                               rocblas_int        batch_count,
                               rocblas_int        lda,
                               host_vector<float> hCPU[],
                               host_vector<float> hGPU[],
                               double             abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(rocblas_int         M,
                               rocblas_int         N,
                               rocblas_int         batch_count,
                               rocblas_int         lda,
                               host_vector<double> hCPU[],
                               host_vector<double> hGPU[],
                               double              abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}