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
#include "FLAME.h"
#include <rocblas.h>

extern "C" {
doublereal dlange_(char* norm, integer* m, integer* n, doublereal* a, integer* lda, doublereal* work);
real slange_(char* norm, integer* m, integer* n, real* a, integer* lda, real* work);
real clange_(char *norm, integer *m, integer *n, complex *a, integer *lda, real *work);
doublereal zlange_(char *norm, integer *m, integer *n, doublecomplex *a, integer *lda, doublereal *work);
}

void flame_saxpy(
    rocblas_int size, float alpha, float* X, rocblas_int incx, float* Y, rocblas_int incy)
{
    bl1_saxpy(size, &alpha, X, incx, Y, incy);
}

void flame_daxpy(
    rocblas_int size, double alpha, double* X, rocblas_int incx, double* Y, rocblas_int incy)
{
    bl1_daxpy(size, &alpha, X, incx, Y, incy);
}

void flame_caxpy(
    rocblas_int size, float alpha, rocblas_float_complex* X, rocblas_int incx, rocblas_float_complex* Y, rocblas_int incy)
{
    bl1_caxpy(size, (scomplex *)&alpha, (scomplex*)X, incx, (scomplex*)Y, incy);
}

void flame_zaxpy(
    rocblas_int size, double alpha, rocblas_double_complex* X, rocblas_int incx, rocblas_double_complex* Y, rocblas_int incy)
{
    bl1_zaxpy(size, (dcomplex *)&alpha, (dcomplex*)X, incx, (dcomplex*)Y, incy);
}

float flame_slange(
    char norm_type, rocblas_int M, rocblas_int N, float* A, rocblas_int lda, float* work)
{
    return slange_(&norm_type, &M, &N, A, &lda, work);
}

double flame_dlange(
    char norm_type, rocblas_int M, rocblas_int N, double* A, rocblas_int lda, double* work)
{
    return dlange_(&norm_type, &M, &N, A, &lda, work);
}

float flame_clange(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_float_complex* A, rocblas_int lda, float* work)
{
    return clange_(&norm_type, &M, &N, (complex*)A, &lda, work);
}

double flame_zlange(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_double_complex* A, rocblas_int lda, double* work)
{
    return zlange_(&norm_type, &M, &N, (doublecomplex*)A, &lda, work);
}