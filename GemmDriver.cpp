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

#include "utility.hpp"
#include <fstream>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef VALIDATE
#include "blis_interface.hpp"
#include "validate.hpp"
#endif

template <typename T>
void BenchGemmStridedBatched(const Arguments& arg, std::promise<std::pair<double,double>> promise)
{
    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    rocblas_stride stride_a    = arg.stride_a;
    rocblas_stride stride_b    = arg.stride_b;
    rocblas_stride stride_c    = arg.stride_c;
    rocblas_int batch_count = arg.batch_count;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_local_handle handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    // Early exit
    if(!M || !N || !batch_count)
        return;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0)
    {
        rocblas_cout << "Invalid sizes...exiting" << std::endl;
        exit(1);
    }

    rocblas_int reinit_c = arg.reinit_c && h_beta != 0;
    rocblas_int time_each_iter = arg.time_each_iter || reinit_c;
    double      host_time;
    double      rocblas_gflops;
    static double cpu_time_used, cblas_gflops;
    int         deviceId;
    if(multi_device>1)
        hipGetDevice(&deviceId);

    double rocblas_error = 0.0;

    size_t size_one_a
        = transA == rocblas_operation_none ? size_t(K) * size_t(lda) : size_t(M) * size_t(lda);
    size_t size_one_b
        = transB == rocblas_operation_none ? size_t(N) * size_t(ldb) : size_t(K) * size_t(ldb);
    size_t size_one_c = N * ldc;

    size_t size_A = size_one_a + size_t(stride_a) * size_t(batch_count - 1);
    size_t size_B = size_one_b + size_t(stride_b) * size_t(batch_count - 1);
    size_t size_C = size_one_c + size_t(stride_c) * size_t(batch_count - 1);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dB(size_B);
    device_vector<T> dC(size_C);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    if((!dA && size_A) || (!dB && size_B) || (!dC && size_C) || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    bool vChecks = (arg.unit_check || arg.norm_check);
    bool transferOutput = (vChecks || storeOutputData);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    static host_vector<T> hA(size_A);
    static host_vector<T> hB(size_B);
    static host_vector<T> hC(size_C);
    host_vector<T> hC_1(transferOutput ? size_C : 0);
    static host_vector<T> hC_gold(vChecks ? size_C : 0);
    static host_vector<T> hC_orig(arg.reinit_c ? size_C : 0);

    // Initial Data on CPU
    if((multi_device>1 && deviceId==0) || multi_device == 1)
    {
        if(arg.initialization == rocblas_initialization_random_int)
        {
            //  Old
            rocblas_seedrand();
            rocblas_init<T>(hA, A_row, A_col, lda, stride_a, batch_count);
            rocblas_init_alternating_sign<T>(hB, B_row, B_col, ldb, stride_b, batch_count);
            if(rocblas_isnan(arg.beta))
                rocblas_init_nan<T>(hC, M, N, ldc, stride_c, batch_count);
            else
                rocblas_init<T>(hC, M, N, ldc, stride_c, batch_count);
        }
        else if(arg.initialization == rocblas_initialization_random_narrow)
        {
            init_narrow_range_random_gemm<T>(transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            hA,
                                            lda,
                                            stride_a,
                                            hB,
                                            ldb,
                                            stride_b,
                                            hC,
                                            ldc,
                                            stride_c,
                                            batch_count);
        }
        else if(arg.initialization == rocblas_initialization_random_broad)
        {
            init_broad_range_random_gemm<T>(transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            hA,
                                            lda,
                                            stride_a,
                                            hB,
                                            ldb,
                                            stride_b,
                                            hC,
                                            ldc,
                                            stride_c,
                                            batch_count);
        }
        else if(arg.initialization == rocblas_initialization_random_full)
        {
            init_full_range_random_gemm<T>(transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        hA,
                                        lda,
                                        stride_a,
                                        hB,
                                        ldb,
                                        stride_b,
                                        hC,
                                        ldc,
                                        stride_c,
                                        batch_count);
        }
        else if(arg.initialization == rocblas_initialization_const)
        {
            init_constant_gemm<T>(transA,
                                transB,
                                M,
                                N,
                                K,
                                hA,
                                lda,
                                stride_a,
                                hB,
                                ldb,
                                stride_b,
                                hC,
                                ldc,
                                stride_c,
                                batch_count,
                                arg.initVal);
        }
        else if(arg.initialization == rocblas_initialization_trig_float)
        {
            rocblas_init_sin<T>(hA, A_row, A_col, lda, stride_a, batch_count);
            rocblas_init_cos<T>(hB, B_row, B_col, ldb, stride_b, batch_count);
            if(rocblas_isnan(arg.beta))
                rocblas_init_nan<T>(hC, M, N, ldc, stride_c, batch_count);
            else
                rocblas_init_sin<T>(hC, M, N, ldc, stride_c, batch_count);
        }
        else if(arg.initialization == rocblas_initialization_hpl)
        {
            rocblas_seedrand();
            rocblas_init_hpl<T>(hA, A_row, A_col, lda, stride_a, batch_count);
            rocblas_init_hpl<T>(hB, B_row, B_col, ldb, stride_b, batch_count);
            if(rocblas_isnan(arg.beta))
                rocblas_init_nan<T>(hC, M, N, ldc, stride_c, batch_count);
            else
                rocblas_init_hpl<T>(hC, M, N, ldc, stride_c, batch_count);
        }
        else if(arg.initialization == rocblas_initialization_file)
        {
            loadFromBin(transA,
                        transB,
                        M,
                        N,
                        K,
                        hA,
                        lda,
                        a_file,
                        hB,
                        ldb,
                        b_file,
                        hC,
                        ldc,
                        c_file,
                        batch_count);
        }
        if(vChecks)
            hC_gold = hC;
        if(reinit_c)
            hC_orig = hC;
        memBarrier.wait();
    }
    else
        memBarrier.wait();

    if(storeInitData)
    {
        storeInitToBin<T,T>(transA, transB, M, N, K, hA, lda, a_file, hB, ldb, b_file, hC, ldc, c_file, batch_count);
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(T) * size_C, hipMemcpyHostToDevice));

#ifdef VALIDATE
    if(arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched<T>(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            &h_alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            &h_beta,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));
        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(T) * size_C, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched<T>(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            d_beta,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));

        if(multi_device > 1 && deviceId!=0)
        {
            memBarrier2.wait(deviceId);
        }

        if(multi_device==1 || (multi_device > 1 && deviceId==0))
        {
            // CPU BLAS
            cpu_time_used = get_time_us();
            for(rocblas_int i = 0; i < batch_count; i++)
            {
                blis_gemm<T>(transA,
                            transB,
                            M,
                            N,
                            K,
                            h_alpha,
                            hA + stride_a * i,
                            lda,
                            hB + stride_b * i,
                            ldb,
                            h_beta,
                            hC_gold + stride_c * i,
                            ldc);
            }
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = gemm_gflop_count<T>(M, N, K) * batch_count / cpu_time_used * 1e6;

            if(multi_device > 1)
            {
                memBarrier2.wait(deviceId);
            }
        }

        //releasing already used host memory
        hA=host_vector<T>();
        hB=host_vector<T>();
        hC=host_vector<T>();

        for(int i=0; i<2; i++)
        {
            if(arg.unit_check)
            {
                if(std::is_same<T, rocblas_half> {} && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, batch_count, ldc, stride_c, hC_gold, hC_1, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, batch_count, ldc, stride_c, hC_gold, hC_1);
                }
            }

            if(arg.norm_check)
            {
                double error
                    = fabs(norm_check_general<T>('F', M, N, ldc, stride_c, batch_count, hC_gold, hC_1));

                rocblas_error = error > rocblas_error ? error : rocblas_error;
            }
            if(i==0)
            {
                CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));
            }
        }
    }
#endif

    int number_cold_calls = 2;
    int number_hot_calls  = arg.iters;
    hipEvent_t start, stop, flush;
    hipEventCreateWithFlags(&flush, hipEventReleaseToSystem);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float kernel_time = 0.0f;
    host_time        = 0.0;
    float kernel_time_iter = 0.0f;
    double host_time_iter = 0.0f;

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    for(int i = 0; i < number_cold_calls; i++)
    {
        rocblas_gemm_strided_batched<T>(handle,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        &h_alpha,
                                        dA,
                                        lda,
                                        stride_a,
                                        dB,
                                        ldb,
                                        stride_b,
                                        &h_beta,
                                        dC,
                                        ldc,
                                        stride_c,
                                        batch_count);
    }

    if(time_each_iter)
    {
        for(int i = 0; i < number_hot_calls; i++)
        {
            if(reinit_c && ((arg.norm_check && i == 0) || i > 0))
                CHECK_HIP_ERROR(hipMemcpy(dC, hC_orig, sizeof(T) * size_C, hipMemcpyHostToDevice));
            if(arg.flush_gpu_cache)
                hipEventRecord(flush, NULL);

            host_time_iter = get_time_us();
            hipEventRecord(start, NULL);

            rocblas_gemm_strided_batched<T>(handle,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        &h_alpha,
                                        dA,
                                        lda,
                                        stride_a,
                                        dB,
                                        ldb,
                                        stride_b,
                                        &h_beta,
                                        dC,
                                        ldc,
                                        stride_c,
                                        batch_count);

            hipEventRecord(stop, NULL);
            hipEventSynchronize(stop);
            host_time += get_time_us() - host_time_iter;
            hipEventElapsedTime(&kernel_time_iter, start, stop);
            kernel_time+=kernel_time_iter;
        }
    }
    else
    {
        std::pair<double,double> times;
        if(multi_device>1)
        {
            usleep(0.5 * 1000000);
            perfBarrier.wait(deviceId);
        }
        times.first = get_time_us(); // in microseconds
        hipEventRecord(start, NULL);
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_strided_batched<T>(handle,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        &h_alpha,
                                        dA,
                                        lda,
                                        stride_a,
                                        dB,
                                        ldb,
                                        stride_b,
                                        &h_beta,
                                        dC,
                                        ldc,
                                        stride_c,
                                        batch_count);
        }

        hipEventRecord(stop, NULL);
        hipEventSynchronize(stop);
        times.second = get_time_us();
        if(multi_device>1)
            promise.set_value(times);
        hipEventElapsedTime(&kernel_time, start, stop);
        host_time = times.second-times.first;
    }

    if(storeOutputData)
    {
        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));
        storeOutputToBin<T>(N, hC_1, ldc, o_file, batch_count);
    }

    rocblas_gflops = gemm_gflop_count<T>(M, N, K) * batch_count * number_hot_calls  / kernel_time * 1e3;

    std::stringstream msg;
    if(multi_device>1)
    {
        double host_gflops = gemm_gflop_count<T>(M, N, K) * number_hot_calls / (host_time) * 1e6;
        msg << "Device " << deviceId << std::endl
        << "transA,transB,M,N,K,alpha,lda,stride_a,ldb,stride_b,beta,ldc,stride_c,Batch_"
            "Count,rocblas-Gflops,rocblas-Gflops(using host_time),host_time(us),kernel_time(us)" << std::endl
        << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << "," << arg.get_alpha<T>() 
        << "," << lda << "," << stride_a << "," << ldb << "," << stride_b << "," << arg.get_beta<T>() 
        << "," << ldc << "," << stride_c << "," << batch_count << "," << rocblas_gflops << "," << host_gflops << "," 
        << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;
    }
    else
        msg << "transA,transB,M,N,K,alpha,lda,stride_a,ldb,stride_b,beta,ldc,stride_c,Batch_"
            "Count,rocblas-Gflops,host_time(us),kernel_time(us)" << std::endl
        << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << "," << arg.get_alpha<T>() 
        << "," << lda << "," << stride_a << "," << ldb << "," << stride_b << "," << arg.get_beta<T>() 
        << "," << ldc << "," << stride_c << "," << batch_count << "," << rocblas_gflops << "," 
        << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;

    if(arg.norm_check)
    {
        msg << "cblas-Gflops,us,rocblas-error" << std::endl
        << cblas_gflops << "," << cpu_time_used << "," << rocblas_error << std::endl;
    }

    rocblas_cout << msg.str();
}

template <typename Ti, typename To, typename Tc>
void BenchGemmEx(Arguments& arg, std::promise<std::pair<double,double>> promise)
{
    rocblas_gemm_algo algo           = static_cast<rocblas_gemm_algo>(arg.algo);
    int32_t           solution_index = arg.solution_index;
    uint32_t          flags          = arg.flags;

    bool nantest = rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai);
    if(!std::is_same<To, float>{} && !std::is_same<To, double>{}
       && !std::is_same<To, rocblas_half>{} && !is_complex<To> && nantest)
        return; // Exclude integers or other types which don't support NaN

    Tc h_alpha_Tc = arg.get_alpha<Tc>();
    Tc h_beta_Tc  = arg.get_beta<Tc>();

    rocblas_int reinit_c = arg.reinit_c && h_beta_Tc != 0;
    rocblas_int c_equals_d = arg.c_equals_d;
    rocblas_int time_each_iter = arg.time_each_iter || reinit_c || arg.flush_gpu_cache;
    rocblas_int tensile_timing = arg.tensile_timing;

    double        host_time;
    double        rocblas_gflops;
    static double cpu_time_used, cblas_gflops;
    double        rocblas_error = 0.0;
    int           deviceId;

    if(multi_device>1)
        hipGetDevice(&deviceId);

    rocblas_local_handle handle;
    auto                 transA = char2rocblas_operation(arg.transA);
    auto                 transB = char2rocblas_operation(arg.transB);
    auto                 M = arg.M, N = arg.N, K = arg.K;
    auto                 lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    auto                 A_row = transA == rocblas_operation_none ? M : K;
    auto                 A_col = transA == rocblas_operation_none ? K : M;
    auto                 B_row = transB == rocblas_operation_none ? K : N;
    auto                 B_col = transB == rocblas_operation_none ? N : K;

    // check for invalid sizes
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || (ldd < M && !c_equals_d)
       || (std::is_same<Ti, int8_t> {}
           && (K % 4 != 0 || (transA != rocblas_operation_none && lda % 4 != 0)
               || (transB == rocblas_operation_none && ldb % 4 != 0))))
    {
        rocblas_cout << "Invalid sizes...exiting" << std::endl;
        exit(1);
    }

    const size_t size_A = size_t(lda) * size_t(A_col);
    const size_t size_B = size_t(ldb) * size_t(B_col);
    const size_t size_C = size_t(ldc) * size_t(N);
    const size_t size_D = c_equals_d ? 0 : size_t(ldd) * size_t(N);

    // allocate memory on device
    device_vector<Ti> dA(size_A);
    device_vector<Ti> dB(size_B);
    device_vector<To> dC(size_C);
    device_vector<To> dD(size_D);
    device_vector<Tc> d_alpha_Tc(1);
    device_vector<Tc> d_beta_Tc(1);
    if(!dA || !dB || !dC || !dD || !d_alpha_Tc || !d_beta_Tc)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    bool vChecks = (arg.unit_check || arg.norm_check);
    bool transferOutput = (vChecks || storeOutputData);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    static host_vector<Ti> hA(size_A);
    static host_vector<Ti> hB(size_B);
    static host_vector<To> hC(size_C);
    host_vector<To> hC_1(transferOutput ? size_C : 0);
    static host_vector<To> hC_gold(vChecks && c_equals_d ? size_C : 0);
    host_vector<To> hD_1(size_D);
    static host_vector<To> hD_gold(vChecks ? size_D : 0);
    static host_vector<To> hC_orig(arg.reinit_c ? size_C : 0);

    if((multi_device>1 && deviceId==0) || multi_device == 1)
    {
        if(arg.initialization == rocblas_initialization_random_int)
        {
            //  Old
            rocblas_seedrand();
            rocblas_init<Ti>(hA, A_row, A_col, lda);
            rocblas_init_alternating_sign<Ti>(hB, B_row, B_col, ldb);
            if(nantest)
                rocblas_init_nan<To>(hC, M, N, ldc);
            else
                rocblas_init<To>(hC, M, N, ldc);
        }
        else if(arg.initialization == rocblas_initialization_random_narrow)
        {
            init_narrow_range_random_gemm<Ti,To>(transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            hA,
                                            lda,
                                            size_A,
                                            hB,
                                            ldb,
                                            size_B,
                                            hC,
                                            ldc,
                                            size_C);
        }
        else if(arg.initialization == rocblas_initialization_random_broad)
        {
            init_broad_range_random_gemm<Ti,To>(transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            hA,
                                            lda,
                                            size_A,
                                            hB,
                                            ldb,
                                            size_B,
                                            hC,
                                            ldc,
                                            size_C);
        }
        else if(arg.initialization == rocblas_initialization_random_full)
        {
            init_full_range_random_gemm<Ti,To>(transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            hA,
                                            lda,
                                            size_A,
                                            hB,
                                            ldb,
                                            size_B,
                                            hC,
                                            ldc,
                                            size_C);
        }
        else if(arg.initialization == rocblas_initialization_const)
        {
            init_constant_gemm<Ti,To>(transA,
                                transB,
                                M,
                                N,
                                K,
                                hA,
                                lda,
                                size_A,
                                hB,
                                ldb,
                                size_B,
                                hC,
                                ldc,
                                size_C,
                                Ti(arg.initVal));
        }
        else if(arg.initialization == rocblas_initialization_trig_float)
        {
            rocblas_init_sin<Ti>(hA, A_row, A_col, lda);
            rocblas_init_cos<Ti>(hB, B_row, B_col, ldb);
            if(rocblas_isnan(arg.beta))
                rocblas_init_nan<To>(hC, M, N, ldc);
            else
                rocblas_init_sin<To>(hC, M, N, ldc);
        }
        else if(arg.initialization == rocblas_initialization_hpl)
        {
            rocblas_seedrand();
            rocblas_init_hpl<Ti>(hA, A_row, A_col, lda);
            rocblas_init_hpl<Ti>(hB, B_row, B_col, ldb);
            if(rocblas_isnan(arg.beta))
                rocblas_init_nan<To>(hC, M, N, ldc);
            else
                rocblas_init_hpl<To>(hC, M, N, ldc);
        }
        else if(arg.initialization == rocblas_initialization_file)
        {
            loadFromBin<Ti,To>(transA, transB, M, N, K, hA, lda, a_file, hB, ldb, b_file, hC, ldc, c_file, 1);
        }

        if(std::is_same<To, rocblas_half>{} && std::is_same<Tc, float>{})
        {
            // half precision IEEE has max and lowest values 65504 and -65504,
            // float precision IEEE has max and lowest values 3.403e+38 and -3.403e+38
            // the following will overflow to inf in half arithmetic,
            // but it will equal zero in float arithmetic   65504 * 2 - 65504 * 2
            //
            // set matrix A and matrix B upper left block to values below to cause
            // inf overflow with 16 bit arithmetic, but no overflow for 32 bit arithmetic
            //
            // 65500 65500             2   -2
            // 65500 65500            -2    2
            //
            const rocblas_half ieee_half_near_max(65504.0 - 4.0);
            const rocblas_half positive_two(2.0);
            const rocblas_half negative_two(-2.0);
            if(M >= 2 && N >= 2 && K >= 2)
            {
                hA[0]       = Ti(ieee_half_near_max);
                hA[1]       = Ti(ieee_half_near_max);
                hA[lda]     = Ti(ieee_half_near_max);
                hA[lda + 1] = Ti(ieee_half_near_max);
                hB[0]       = Ti(positive_two);
                hB[1]       = Ti(negative_two);
                hB[ldb]     = Ti(negative_two);
                hB[ldb + 1] = Ti(positive_two);
            }
        }
        else if(std::is_same<Ti, rocblas_bfloat16>{} && std::is_same<Tc, float>{})
        {
            // half precision IEEE has max and lowest values 65504 and -65504,
            // float precision IEEE has max and lowest values 3.403e+38 and -3.403e+38
            // the following will overflow to inf in half arithmetic,
            // but it will equal zero in float arithmetic   65504 * 2 - 65504 * 2
            //
            // set matrix A and matrix B upper left block to values below to cause
            // inf overflow with 16 bit arithmetic, but no overflow for 32 bit arithmetic
            //
            // 65500 65500             2   -2
            // 65500 65500            -2    2
            //
            const float ieee_half_near_max = 65504.0f - 4.0f;
            const float positive_two       = 2.0f;
            const float negative_two       = -2.0f;
            if(M >= 2 && N >= 2 && K >= 2)
            {
                hA[0]       = Ti(ieee_half_near_max);
                hA[1]       = Ti(ieee_half_near_max);
                hA[lda]     = Ti(ieee_half_near_max);
                hA[lda + 1] = Ti(ieee_half_near_max);
                hB[0]       = Ti(positive_two);
                hB[1]       = Ti(negative_two);
                hB[ldb]     = Ti(negative_two);
                hB[ldb + 1] = Ti(positive_two);
            }
        }

        if(!c_equals_d)
            rocblas_init<To>(hD_1, M, N, ldd);

        if(vChecks)
        {
            if(!c_equals_d)
                hD_gold = hD_1;
            else
                hC_gold = hC;
        }
        if(reinit_c)
            hC_orig = hC;
             
        memBarrier.wait();
    }
    else
        memBarrier.wait();

    if(storeInitData)
    {
        storeInitToBin<Ti,To>(transA, transB, M, N, K, hA, lda, a_file, hB, ldb, b_file, hC, ldc, c_file, 1);
    }

    // copy data from CPU to device
    // if int8 and A not transposed and valid case, pack A
    if(std::is_same<Ti, int8_t> {} && transA == rocblas_operation_none)
    {
        host_vector<Ti> hA_packed(hA);

        rocblas_packInt8(hA_packed, M, K, lda);
        CHECK_HIP_ERROR(hipMemcpy(dA, hA_packed, sizeof(Ti) * size_A, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ti) * size_A, hipMemcpyHostToDevice));
    }

    // if int8 and B transposed and valid case, pack B
    if(std::is_same<Ti, int8_t> {} && transB != rocblas_operation_none)
    {
        host_vector<Ti> hB_packed(hB);

        rocblas_packInt8(hB_packed, N, K, ldb);
        CHECK_HIP_ERROR(hipMemcpy(dB, hB_packed, sizeof(Ti) * size_B, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(Ti) * size_B, hipMemcpyHostToDevice));
    }

    CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(To) * size_C, hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(hipMemcpy(dD, hD_1, sizeof(To) * size_D, hipMemcpyHostToDevice));

#ifdef VALIDATE
    if(arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            &h_alpha_Tc,
                                            dA,
                                            arg.a_type,
                                            lda,
                                            dB,
                                            arg.b_type,
                                            ldb,
                                            &h_beta_Tc,
                                            dC,
                                            arg.c_type,
                                            ldc,
                                            c_equals_d ? dC : dD,
                                            c_equals_d ? arg.c_type : arg.d_type,
                                            c_equals_d ? ldc : ldd,
                                            arg.compute_type,
                                            algo,
                                            solution_index,
                                            flags));                     

        if(c_equals_d)
            CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(To) * size_C, hipMemcpyDeviceToHost));
        else
            CHECK_HIP_ERROR(hipMemcpy(hD_1, dD, sizeof(To) * size_D, hipMemcpyDeviceToHost));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));

        if(c_equals_d)
            CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(To) * size_C, hipMemcpyHostToDevice));
        else
            CHECK_HIP_ERROR(hipMemcpy(dD, hD_gold, sizeof(To) * size_D, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            d_alpha_Tc,
                                            dA,
                                            arg.a_type,
                                            lda,
                                            dB,
                                            arg.b_type,
                                            ldb,
                                            d_beta_Tc,
                                            dC,
                                            arg.c_type,
                                            ldc,
                                            c_equals_d ? dC : dD,
                                            c_equals_d ? arg.c_type : arg.d_type,
                                            c_equals_d ? ldc : ldd,
                                            arg.compute_type,
                                            algo,
                                            solution_index,
                                            flags));

        if(multi_device > 1 && deviceId!=0)
        {
            memBarrier2.wait(deviceId);
        }

        if(multi_device==1 || (multi_device > 1 && deviceId==0))
        {
            // CPU BLAS
            // copy C matrix into D matrix
            if(!c_equals_d)
            {
                for(int i2 = 0; i2 < N; i2++)
                {
                    for(int i1 = 0; i1 < M; i1++)
                    {
                        hD_gold[i1 + i2 * ldd] = hC[i1 + i2 * ldc];
                    }
                }
            }
            cpu_time_used = get_time_us();
            blis_gemm<Ti,To,Tc>(
                transA, transB, M, N, K, h_alpha_Tc, hA, lda, hB, ldb, h_beta_Tc, c_equals_d ? hC_gold : hD_gold, c_equals_d ? ldc : ldd);
            //if C does not equal D check if C changed

            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = gemm_gflop_count<To>(M, N, K) / cpu_time_used * 1e6;

            if(multi_device > 1)
            {
                memBarrier2.wait(deviceId);
            }
        }

        //releasing already used host memory
        hA=host_vector<Ti>();
        hB=host_vector<Ti>();
        hC=host_vector<To>();

        for(int i=0; i<2; i++)
        {
            if(arg.unit_check)
            {
                if(std::is_same<Tc, rocblas_half> {} && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<Tc>;
                    if(!c_equals_d)
                    {
                        near_check_general<To>(M, N, ldd, hD_gold, hD_1, tol);
                    }
                    else
                    {
                        unit_check_general<To>(M, N, ldc, hC_gold, hC_1);
                    }
                }
                else
                {
                    if(!c_equals_d)
                    {
                        unit_check_general<To>(M, N, ldd, hD_gold, hD_1);
                    }
                    else
                    {
                        unit_check_general<To>(M, N, ldc, hC_gold, hC_1);
                    }
                }
            }

            if(arg.norm_check)
            {
                auto err = 0.0;
                if(!c_equals_d)
                {
                    err = fabs(norm_check_general<To>('F', M, N, ldd, hD_gold, hD_1));
                }
                else
                {
                    err = fabs(norm_check_general<To>('F', M, N, ldc, hC_gold, hC_1));
                }

                rocblas_error = err > rocblas_error ? err : rocblas_error;
            }

            if(i==0)
            {
                if(c_equals_d)
                    CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(To) * size_C, hipMemcpyDeviceToHost));
                else
                    CHECK_HIP_ERROR(hipMemcpy(hD_1, dD, sizeof(To) * size_D, hipMemcpyDeviceToHost));
            }
        }
    }
#endif

    int number_cold_calls = 2;
    int number_hot_calls  = arg.iters;
    int numEvents = (tensile_timing ? number_hot_calls + 1: 1);

    hipEvent_t flush, start[numEvents], stop[numEvents];
    hipEventCreateWithFlags(&flush, hipEventReleaseToSystem);

    for(int i =0; i < numEvents;i++)
    {
        hipEventCreate(&start[i]);
        hipEventCreate(&stop[i]);
    }

    float kernel_time = 0.0f;
    float tensile_time = 0.0f;
    host_time        = 0.0;
    float kernel_time_iter = 0.0f;
    double host_time_iter = 0.0f;
    To* output_pointer = c_equals_d ? dC : dD;
    rocblas_datatype output_type = c_equals_d ? arg.c_type : arg.d_type;
    rocblas_int ld_output = c_equals_d ? ldc : ldd;

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    for(int i = 0; i < number_cold_calls; i++)
    {
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            &h_alpha_Tc,
                                            dA,
                                            arg.a_type,
                                            lda,
                                            dB,
                                            arg.b_type,
                                            ldb,
                                            &h_beta_Tc,
                                            dC,
                                            arg.c_type,
                                            ldc,
                                            output_pointer,
                                            output_type,
                                            ld_output,
                                            arg.compute_type,
                                            algo,
                                            solution_index,
                                            flags));
    }

    if(time_each_iter)
    {
        for(int i = 0; i < number_hot_calls; i++)
        {
            if(reinit_c && ((arg.norm_check && i == 0) || i > 0))
                CHECK_HIP_ERROR(hipMemcpy(dC, hC_orig, sizeof(To) * size_C, hipMemcpyHostToDevice));
            if(arg.flush_gpu_cache)
                hipEventRecord(flush, NULL);

            host_time_iter = get_time_us();
            hipEventRecord(start[numEvents-1], NULL);

            rocblas_gemm_ex(handle,
                            transA,
                            transB,
                            M,
                            N,
                            K,
                            &h_alpha_Tc,
                            dA,
                            arg.a_type,
                            lda,
                            dB,
                            arg.b_type,
                            ldb,
                            &h_beta_Tc,
                            dC,
                            arg.c_type,
                            ldc,
                            output_pointer,
                            output_type,
                            ld_output,
                            arg.compute_type,
                            algo,
                            solution_index,
                            flags);

            hipEventRecord(stop[numEvents-1], NULL);
            hipEventSynchronize(stop[numEvents-1]);
            host_time += get_time_us() - host_time_iter;
            hipEventElapsedTime(&kernel_time_iter, start[numEvents-1], stop[numEvents-1]);
            kernel_time+=kernel_time_iter;
        }
    }
    else
    {
        std::pair<double,double> times;
        if(multi_device>1)
        {
            usleep(0.5 * 1000000);
            perfBarrier.wait(deviceId);
        }
        times.first = get_time_us(); // in microseconds
        hipEventRecord(start[numEvents-1], NULL);
        for(int i = 0; i < number_hot_calls; i++)
        {
            ROCBLAS_INVOKE_START_STOP_EVENTS(handle, tensile_timing ? start[i]: nullptr, tensile_timing ? stop[i] : nullptr,
            rocblas_gemm_ex(handle,
                            transA,
                            transB,
                            M,
                            N,
                            K,
                            &h_alpha_Tc,
                            dA,
                            arg.a_type,
                            lda,
                            dB,
                            arg.b_type,
                            ldb,
                            &h_beta_Tc,
                            dC,
                            arg.c_type,
                            ldc,
                            output_pointer,
                            output_type,
                            ld_output,
                            arg.compute_type,
                            algo,
                            solution_index,
                            flags));
        }

        hipEventRecord(stop[numEvents-1], NULL);
        hipEventSynchronize(stop[numEvents-1]);

        times.second = get_time_us();
        if(multi_device>1)
            promise.set_value(times);
        host_time = times.second-times.first;
        for(int i=0; i<numEvents-1;i++)
        {
            hipEventElapsedTime(&kernel_time_iter, start[i], stop[i]);
            tensile_time+=kernel_time_iter;
        }

        hipEventElapsedTime(&kernel_time, start[numEvents-1], stop[numEvents-1]);
    }

    if(storeOutputData)
    {
        CHECK_HIP_ERROR(hipMemcpy(c_equals_d ? hC_1 : hD_1, output_pointer, sizeof(To) * ld_output * N, hipMemcpyDeviceToHost));
        storeOutputToBin<To>(N, c_equals_d ? hC_1 : hD_1, ld_output, o_file, 1);
    }

    rocblas_gflops = gemm_gflop_count<Ti>(M, N, K) * number_hot_calls / (tensile_timing ? tensile_time : kernel_time) * 1e3;
    double host_gflops = gemm_gflop_count<Ti>(M, N, K) * number_hot_calls / (host_time) * 1e6;

    std::stringstream msg;
    if(tensile_timing)
    {
        if(multi_device>1)
        {
            msg << "Device " << deviceId << std::endl
            << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,rocblas-Gflops(using host_time),host_time(us),kernel_time(us)" 
            << ",tensile_time(us)" << std::endl << rocblas2char_operation(transA) << "," << rocblas2char_operation(transB) << ","
            << M << "," << N << "," << K << "," << arg.alpha << "," << lda << "," << ldb
            << "," << arg.beta << "," << ldc << "," << rocblas_gflops << "," << host_gflops << ","
            << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << ","
            << tensile_time/number_hot_calls*1000 << std::endl;
        }
        else
            msg << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,host_time(us),kernel_time(us)" 
            << ",tensile_time(us)" << std::endl << rocblas2char_operation(transA) << "," << rocblas2char_operation(transB) << ","
            << M << "," << N << "," << K << "," << arg.alpha << "," << lda << "," << ldb
            << "," << arg.beta << "," << ldc << "," << rocblas_gflops << ","
            << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << ","
            << tensile_time/number_hot_calls*1000 << std::endl;    
    }
    else
    {
        if(multi_device>1)
        {
            msg << "Device " << deviceId << std::endl
            << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,rocblas-Gflops(using host_time),host_time(us),kernel_time(us)"<< std::endl
            << rocblas2char_operation(transA) << "," << rocblas2char_operation(transB) << ","
            << M << "," << N << "," << K << "," << arg.alpha << "," << lda << "," << ldb
            << "," << arg.beta << "," << ldc << "," << rocblas_gflops << "," << host_gflops << ","
            << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;
        }
        else
            msg << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,host_time(us),kernel_time(us)"<< std::endl
                << rocblas2char_operation(transA) << "," << rocblas2char_operation(transB) << ","
            << M << "," << N << "," << K << "," << arg.alpha << "," << lda << "," << ldb
            << "," << arg.beta << "," << ldc << "," << rocblas_gflops  << ","
            << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;
    }

    if(arg.norm_check)
    {
        msg << "cblas-Gflops,us,rocblas-error" << std::endl
        << cblas_gflops << "," << cpu_time_used << "," << rocblas_error << std::endl;
    }

    rocblas_cout << msg.str();
}

template <typename T>
void BenchGemm(Arguments& arg, std::promise<std::pair<double,double>> promise)
{
    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    rocblas_int          reinit_c = arg.reinit_c && h_beta != 0;
    rocblas_int          time_each_iter = arg.time_each_iter || reinit_c;
    double               host_time;
    double               rocblas_gflops;
    static double        cblas_gflops, cpu_time_used;
    double               rocblas_error = 0.0;
    rocblas_local_handle handle;
    int deviceId;
    if(multi_device>1)
        hipGetDevice(&deviceId);

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M)
    {
        rocblas_cout << "Invalid sizes...exiting" << std::endl;
        exit(1);
    }

    const auto size_A = size_t(lda) * size_t(A_col);
    const auto size_B = size_t(ldb) * size_t(B_col);
    const auto size_C = size_t(ldc) * size_t(N);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dB(size_B);
    device_vector<T> dC(size_C);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    if(!dA || !dB || !dC || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    bool vChecks = (arg.unit_check || arg.norm_check);
    bool transferOutput = (vChecks || storeOutputData);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    static host_vector<T> hA(size_A);
    static host_vector<T> hB(size_B);
    static host_vector<T> hC(size_C);
    host_vector<T> hC_1(transferOutput ? size_C : 0);
    static host_vector<T> hC_gold(vChecks ? size_C : 0);
    static host_vector<T> hC_orig(arg.reinit_c ? size_C : 0);
    // Initial Data on CPU
    if((multi_device>1 && deviceId==0) || multi_device == 1)
    {
        if(arg.initialization == rocblas_initialization_random_int)
        {
            //  Old
            rocblas_seedrand();
            rocblas_init<T>(hA, A_row, A_col, lda);
            rocblas_init_alternating_sign<T>(hB, B_row, B_col, ldb);
            if(rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai))
                rocblas_init_nan<T>(hC, M, N, ldc);
            else
                rocblas_init<T>(hC, M, N, ldc);
        }
        else if(arg.initialization == rocblas_initialization_random_narrow)
        {
            init_narrow_range_random_gemm<T>(
                transA, transB, M, N, K, hA, lda, size_A, hB, ldb, size_B, hC, ldc, size_C);
        }
        else if(arg.initialization == rocblas_initialization_random_broad)
        {
            init_broad_range_random_gemm<T>(
                transA, transB, M, N, K, hA, lda, size_A, hB, ldb, size_B, hC, ldc, size_C);
        }
        else if(arg.initialization == rocblas_initialization_random_full)
        {
            init_full_range_random_gemm<T>(
                transA, transB, M, N, K, hA, lda, size_A, hB, ldb, size_B, hC, ldc, size_C);
        }
        else if(arg.initialization == rocblas_initialization_const)
        {
            init_constant_gemm<T>(transA,
                                transB,
                                M,
                                N,
                                K,
                                hA,
                                lda,
                                size_A,
                                hB,
                                ldb,
                                size_B,
                                hC,
                                ldc,
                                size_C,
                                T(arg.initVal));
        }
        else if(arg.initialization == rocblas_initialization_trig_float)
        {
            rocblas_init_sin<T>(hA, A_row, A_col, lda);
            rocblas_init_cos<T>(hB, B_row, B_col, ldb);
            if(rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai))
                rocblas_init_nan<T>(hC, M, N, ldc);
            else
                rocblas_init_sin<T>(hC, M, N, ldc);
        }
        else if(arg.initialization == rocblas_initialization_hpl)
        {
            rocblas_seedrand();
            rocblas_init_hpl<T>(hA, A_row, A_col, lda);
            rocblas_init_hpl<T>(hB, B_row, B_col, ldb);
            if(rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai))
                rocblas_init_nan<T>(hC, M, N, ldc);
            else
                rocblas_init_hpl<T>(hC, M, N, ldc);
        }
        else if(arg.initialization == rocblas_initialization_file)
        {
            loadFromBin(
                transA, transB, M, N, K, hA, lda, a_file, hB, ldb, b_file, hC, ldc, c_file, 1);
        }

        if(reinit_c)
            hC_orig = hC;

        if(vChecks)
            hC_gold = hC;
        memBarrier.wait();
    }
    else
    {
        memBarrier.wait();
    }

    if(storeInitData)
    {
        storeInitToBin<T,T>(transA, transB, M, N, K, hA, lda, a_file, hB, ldb, b_file, hC, ldc, c_file, 1);
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(T) * size_C, hipMemcpyHostToDevice));

#ifdef VALIDATE
    if(arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(T) * size_C, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(
            handle, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

        if(multi_device > 1 && deviceId!=0)
        {
            memBarrier2.wait(deviceId);
        }

        if(multi_device==1 || (multi_device > 1 && deviceId==0))
        {
            cpu_time_used = get_time_us();
            
            blis_gemm<T>(transA,
                        transB,
                        M,
                        N,
                        K,
                        h_alpha,
                        hA.data(),
                        lda,
                        hB.data(),
                        ldb,
                        h_beta,
                        hC_gold.data(),
                        ldc);

            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = gemm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;

            if(multi_device > 1)
            {
                memBarrier2.wait(deviceId);
            }
        }

        //releasing already used host memory
        hA=host_vector<T>();
        hB=host_vector<T>();
        hC=host_vector<T>();

        for(int i = 0; i<2; i++)
        {
            if(arg.unit_check)
            {
                if(std::is_same<T, rocblas_half>{} && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, hC_gold, hC_1, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
                }
            }

            if(arg.norm_check)
            {
                auto err1     = fabs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1));
                rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
            }

            if(i==0)
                CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));
        }
    }
#endif

    int number_cold_calls = 2;
    int number_hot_calls  = arg.iters;
    hipEvent_t start, stop, flush;
    hipEventCreateWithFlags(&flush, hipEventReleaseToSystem);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float kernel_time = 0.0f;
    host_time        = 0.0;
    float kernel_time_iter = 0.0f;
    double host_time_iter = 0.0f;

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    for(int i = 0; i < number_cold_calls; i++)
    {
        rocblas_gemm<T>(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);
    }
    

    if(time_each_iter)
    {
        for(int i = 0; i < number_hot_calls; i++)
        {
            if(reinit_c && ((arg.norm_check && i == 0) || i > 0))
                CHECK_HIP_ERROR(hipMemcpy(dC, hC_orig, sizeof(T) * size_C, hipMemcpyHostToDevice));
            if(arg.flush_gpu_cache)
                hipEventRecord(flush, NULL);

            host_time_iter = get_time_us();
            hipEventRecord(start, NULL);

            rocblas_gemm<T>(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);

            hipEventRecord(stop, NULL);
            hipEventSynchronize(stop);
            host_time += get_time_us() - host_time_iter;
            hipEventElapsedTime(&kernel_time_iter, start, stop);
            kernel_time+=kernel_time_iter;
        }
    }
    else
    {
        std::pair<double,double> times;
        if(multi_device>1)
        {
            usleep(0.5 * 1000000);
            perfBarrier.wait(deviceId);
        }
        times.first = get_time_us(); // in microseconds
        hipEventRecord(start, NULL);
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm<T>(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);
        }

        hipEventRecord(stop, NULL);
        hipEventSynchronize(stop);
        times.second = get_time_us();
        if(multi_device>1)
            promise.set_value(times);
        hipEventElapsedTime(&kernel_time, start, stop);
        host_time = times.second-times.first;
    }

    if(storeOutputData)
    {
        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));
        storeOutputToBin<T>(N, hC_1, ldc, o_file, 1);
    }

    rocblas_gflops = gemm_gflop_count<T>(M, N, K) * number_hot_calls / kernel_time * 1e3;

    std::stringstream msg;

    if(multi_device>1)
    {
        double host_gflops = gemm_gflop_count<T>(M, N, K) * number_hot_calls / (host_time) * 1e6;
        msg << "Device " << deviceId << std::endl
        << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,rocblas-Gflops(using host_time),host_time(us),kernel_time(us)" << std::endl
        << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << ","
        << arg.get_alpha<T>() << "," << lda << "," << ldb << "," << arg.get_beta<T>() << "," << ldc
        << "," << rocblas_gflops << "," << host_gflops << "," << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;
    }
    else
        msg << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,host_time(us),kernel_time(us)" << std::endl
        << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << ","
        << arg.get_alpha<T>() << "," << lda << "," << ldb << "," << arg.get_beta<T>() << "," << ldc
        << "," << rocblas_gflops << "," << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;

    if(arg.norm_check)
    {
        msg << "cblas-Gflops,us,rocblas-error" << std::endl
        << cblas_gflops << "," << cpu_time_used << "," << rocblas_error << std::endl;
    }

    rocblas_cout  << msg.str();
}

int launch_bench(Arguments& arg, std::promise<std::pair<double,double>> promise)
{
    if(function == "gemm")
    {
        if(precision == "f32_r" || precision == "s")
        {
            BenchGemm<float>(arg, std::move(promise));
        }
        else if(precision == "f64_r" || precision == "d")
        {
            BenchGemm<double>(arg, std::move(promise));
        }
        else if(precision == "f16_r")
        {
            BenchGemm<rocblas_half>(arg, std::move(promise));
        }
        else
        {
            rocblas_cout << "Precision not implemented, exiting";
            return rocblas_status_not_implemented;
        }
    }
    else if(function == "gemm_strided_batched")
    {
        if(precision == "f32_r" || precision == "s")
        {
            BenchGemmStridedBatched<float>(arg, std::move(promise));
        }
        else if(precision == "f64_r" || precision == "d")
        {
            BenchGemmStridedBatched<double>(arg, std::move(promise));
        }
        else if(precision == "f16_r")
        {
            BenchGemmStridedBatched<rocblas_half>(arg, std::move(promise));
        }
        else
        {
            rocblas_cout << "Precision not implemented, exiting";
            return rocblas_status_not_implemented;
        }
    }
    else if(function == "gemm_ex")
    {
        if((a_type == "f64_r" || a_type == "d") && (b_type == "f64_r" || b_type == "d")
           && (c_type == "f64_r" || c_type == "d") && (d_type == "f64_r" || d_type == "d")
           && (compute_type == "f64_r" || compute_type == "d"))
        {   
            BenchGemmEx<double, double, double>(arg, std::move(promise));
        }
        else if((a_type == "f32_r" || a_type == "s") && (b_type == "f32_r" || b_type == "s")
                && (c_type == "f32_r" || c_type == "s") && (d_type == "f32_r" || d_type == "s")
                && (compute_type == "f32_r" || compute_type == "s"))
        {
            BenchGemmEx<float, float, float>(arg, std::move(promise));
        }
        else if((a_type == "bf16_r") && (b_type == "bf16_r")
                && (c_type == "bf16_r") && (d_type == "bf16_r")
                && (compute_type == "f32_r" || compute_type == "s"))
        {
            BenchGemmEx<rocblas_bfloat16, rocblas_bfloat16, float>(arg, std::move(promise));
        }
        else if(a_type == "f16_r"  && b_type == "f16_r"
                && c_type == "f16_r" && d_type == "f16_r"
                && compute_type == "f16_r")
        {
            BenchGemmEx<rocblas_half, rocblas_half, rocblas_half>(arg, std::move(promise));
        }
        else if(a_type == "f16_r"  && b_type == "f16_r"
                && c_type == "f16_r" && d_type == "f16_r"
                && (compute_type == "f32_r" || compute_type == "s"))
        {
            BenchGemmEx<rocblas_half, rocblas_half, float>(arg, std::move(promise));
        }
        else
        {
            rocblas_cout << "Precision not implemented, exiting";
            return rocblas_status_not_implemented;
        }
    }
    else
    {
        rocblas_cout << "Function not implemented, exiting";
        return rocblas_status_not_implemented;
    }

    return 0;
}

int main(int argc, char* argv[])
{

    Arguments arg;
    readArgs(argc, argv, arg);

    if(arg.norm_check || arg.unit_check)
    {
#ifdef VALIDATE
        setup_blis();
#else
        rocblas_cout << "run ./install -v 1 to enable validation" << std::endl;
        exit(1);
#endif
    }

    auto promise = std::make_unique<std::promise<std::pair<double,double>>[]>(multi_device);

    if(multi_device>1)
    {
        std::vector<std::thread> threads;
        auto future  = std::make_unique<std::future<std::pair<double,double>>[]>(multi_device);

        for(size_t i = 0; i < multi_device; ++i)
            future[i] = promise[i].get_future();

        for(int i = 0 ; i<multi_device; ++i)
            threads.push_back(std::thread([&, i] { set_device(i); launch_bench(arg, std::move(promise[i])); }));

        perfBarrier.wait_to_trigger();

        std::vector<std::pair<double,double>> times(multi_device);

        //wait for promises
        for(size_t i = 0; i < multi_device; ++i)
            times[i] = future[i].get(); 

        double start=times[0].first;
        double end=times[0].second;

        for(int i =0; i<multi_device; i++)
        {
            if(times[i].first < start)
                start = times[i].first;
            if(times[i].second > end)
                end = times[i].second;
        }

        for(int i =0; i<multi_device; i++)
            threads[i].join();

        //print overall run data
        double overall_time = (end-start)/arg.iters;

        double overall_gflops;
        if(arg.d_type == rocblas_datatype_f16_r)
            overall_gflops = gemm_gflop_count<rocblas_half>(arg.M, arg.N, arg.K);
        else if(arg.d_type == rocblas_datatype_bf16_r)
            overall_gflops = gemm_gflop_count<rocblas_bfloat16>(arg.M, arg.N, arg.K);
        else if(arg.d_type == rocblas_datatype_f32_r)
            overall_gflops = gemm_gflop_count<float>(arg.M, arg.N, arg.K);
        else if(arg.d_type == rocblas_datatype_f64_r)
            overall_gflops = gemm_gflop_count<double>(arg.M, arg.N, arg.K);
        else
        {
            rocblas_cout << "Precision not implemented, exiting";
            return rocblas_status_not_implemented;
        }
        overall_gflops /= overall_time / 1e6 / multi_device; 

        rocblas_cout<<"Overall performance using host timing"<<std::endl
        << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,host_time(us)"<< std::endl
        << arg.transA << "," << arg.transB << ","
        << arg.M << "," << arg.N << "," << arg.K << "," << arg.alpha << "," << arg.lda << "," << arg.ldb
        << "," << arg.beta << "," << arg.ldc  << "," << overall_gflops << ","
        << overall_time  << std::endl;
    }
    else
    {
        return launch_bench(arg, std::move(promise[0]));
    }


    return 0;
}
