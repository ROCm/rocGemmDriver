#include "rocblas.h"

void setup_blis();

void blis_dgemm(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int       m,
                rocblas_int       n,
                rocblas_int       k,
                double            alpha,
                double*           A,
                rocblas_int       lda,
                double*           B,
                rocblas_int       ldb,
                double            beta,
                double*           C,
                rocblas_int       ldc);

void blis_sgemm(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int       m,
                rocblas_int       n,
                rocblas_int       k,
                float             alpha,
                float*            A,
                rocblas_int       lda,
                float*            B,
                rocblas_int       ldb,
                float             beta,
                float*            C,
                rocblas_int       ldc);

void blis_bfgemm(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int       m,
                rocblas_int       n,
                rocblas_int       k,
                float             alpha,
                rocblas_bfloat16*            A,
                rocblas_int       lda,
                rocblas_bfloat16*            B,
                rocblas_int       ldb,
                float             beta,
                rocblas_bfloat16*            C,
                rocblas_int       ldc);

template <typename Ti, typename To = Ti, typename Tc = Ti>
void (*blis_gemm)(rocblas_operation transA,
                  rocblas_operation transB,
                  rocblas_int       m,
                  rocblas_int       n,
                  rocblas_int       k,
                  Tc                 alpha,
                  Ti*                A,
                  rocblas_int       lda,
                  Ti*                B,
                  rocblas_int       ldb,
                  Tc                 beta,
                  To*                C,
                  rocblas_int       ldc);

template <>
static constexpr auto blis_gemm<float, float, float> = blis_sgemm;

template <>
static constexpr auto blis_gemm<double, double, double> = blis_dgemm;

// template <typename Ti, typename To, typename Tc>
// void blis_gemm(rocblas_operation transA,
//                   rocblas_operation transB,
//                   rocblas_int       m,
//                   rocblas_int       n,
//                   rocblas_int       k,
//                   Tc                 alpha,
//                   Ti*                A,
//                   rocblas_int       lda,
//                   Ti*                B,
//                   rocblas_int       ldb,
//                   Tc                 beta,
//                   To*                C,
//                   rocblas_int       ldc);

template <>
static constexpr auto blis_gemm<rocblas_bfloat16,rocblas_bfloat16,float> = blis_bfgemm;
