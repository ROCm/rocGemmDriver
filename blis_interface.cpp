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
 
#include <iostream>
#include "blis_interface.hpp"
#include "blis.h"
#include "omp.h"
#include <vector>

trans_t blis_transpose(rocblas_operation trans)
{
    if(trans == rocblas_operation_none)
    {
        return BLIS_CONJ_NO_TRANSPOSE;
    }
    else if(trans == rocblas_operation_transpose)
    {
        return BLIS_CONJ_TRANSPOSE;
    }
    else if(trans == rocblas_operation_conjugate_transpose)
    {
        return BLIS_CONJ_TRANSPOSE;
    }
    else
    {
        std::cerr << "rocblas ERROR: trans != N, T, C" << std::endl;
        exit(1);
    }
}

void setup_blis()
{
    bli_init();
    bli_thread_set_num_threads(omp_get_max_threads());
}

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
                rocblas_int       ldc)
{

    bli_dgemm(blis_transpose(transA),
              blis_transpose(transB),
              m,
              n,
              k,
              (double*)&alpha,
              (double*)A,
              1,
              lda,
              (double*)B,
              1,
              ldb,
              (double*)&beta,
              (double*)C,
              1,
              ldc);
}

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
                rocblas_int       ldc)
{
    bli_sgemm(blis_transpose(transA),
              blis_transpose(transB),
              m,
              n,
              k,
              (float*)&alpha,
              (float*)A,
              1,
              lda,
              (float*)B,
              1,
              ldb,
              (float*)&beta,
              (float*)C,
              1,
              ldc);
}

void blis_hgemm(rocblas_operation transA,
                                                           rocblas_operation transB,
                                                           rocblas_int       m,
                                                           rocblas_int       n,
                                                           rocblas_int       k,
                                                           float             alpha,
                                                           rocblas_half* A,
                                                           rocblas_int       lda,
                                                           rocblas_half* B,
                                                           rocblas_int       ldb,
                                                           float             beta,
                                                           rocblas_half* C,
                                                           rocblas_int       ldc)
{
    // cblas does not support rocblas_half, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    std::vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = static_cast<float>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    bli_sgemm(blis_transpose(transA),
              blis_transpose(transB),
              m,
              n,
              k,
              (float*)&alpha,
              A_float.data(),
              1,
              lda,
              B_float.data(),
              1,
              ldb,
              (float*)&beta,
              C_float.data(),
              1,
              ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = rocblas_half(C_float[i]);
}

void blis_bfgemm(rocblas_operation transA,
                                                           rocblas_operation transB,
                                                           rocblas_int       m,
                                                           rocblas_int       n,
                                                           rocblas_int       k,
                                                           float             alpha,
                                                           rocblas_bfloat16* A,
                                                           rocblas_int       lda,
                                                           rocblas_bfloat16* B,
                                                           rocblas_int       ldb,
                                                           float             beta,
                                                           rocblas_bfloat16* C,
                                                           rocblas_int       ldc)
{
    // cblas does not support rocblas_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    std::vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = static_cast<float>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    bli_sgemm(blis_transpose(transA),
              blis_transpose(transB),
              m,
              n,
              k,
              (float*)&alpha,
              A_float.data(),
              1,
              lda,
              B_float.data(),
              1,
              ldb,
              (float*)&beta,
              C_float.data(),
              1,
              ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<rocblas_bfloat16>(C_float[i]);
}
