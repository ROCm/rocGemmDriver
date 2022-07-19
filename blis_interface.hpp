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

void blis_hgemm(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int       m,
                rocblas_int       n,
                rocblas_int       k,
                float             alpha,
                rocblas_half*            A,
                rocblas_int       lda,
                rocblas_half*            B,
                rocblas_int       ldb,
                float             beta,
                rocblas_half*            C,
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

template <>
static constexpr auto blis_gemm<rocblas_half,rocblas_half,float> = blis_hgemm;
