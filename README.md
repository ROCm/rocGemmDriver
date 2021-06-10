# GemmDriver
Tool to measure GEMM performance, using rocBLAS library.

Getting Started
---------------

The install script gathers and builds neccessary libraries depending on whether validation is needed. This is specified via the -v flag by the user as follows:
```
$ ./install -v 1
```
Note: A user may choose to point to a local copy of rocblas by using the -r (--rocblas) flag and specifying the base rocblas directory.

Usage 
---------------
Gemm parameters are specified via command line arguments. Here is a brief overview of the required arguments and default values for different initialization types.
Example
```
$ ./GemmDriver -f gemm -r s --transposeA N --transposeB N -m 128 -n 128 -k 128 --alpha 1 --lda 128 --ldb 128 --beta 0 --ldc 128 -v 1 --initialization rand_broad
```
The following arguments are the basic parameters for all GEMM launches:
```
-f [ --function ] arg (=gemm)      GEMM function to test. (gemm,
                                   gemm_strided_batched and gemm_ex
-r [ --precision ] arg (=f32_r)   Specifies the input/output precision
                                  Options: s,d,f16_r,bf16_r,f32_r,f64_r
--transposeA arg (=N)              N = no transpose, T = transpose, C =
                                   conjugate transpose
--transposeB arg (=N)              N = no transpose, T = transpose, C =
                                   conjugate transpose
-m [ --sizem ] arg (=128)          Specific matrix size: sizem is only
                                   applicable to BLAS-2 & BLAS-3: the number of
                                   rows or columns in matrix.
-n [ --sizen ] arg (=128)          Specific matrix/vector size: BLAS-1: the
                                   length of the vector. BLAS-2 & BLAS-3: the
                                   number of rows or columns in matrix
-k [ --sizek ] arg (=128)          Specific matrix size:sizek is only
                                   applicable to BLAS-3: the number of columns
                                   in A and rows in B.
--lda arg (=128)                   On entry, LDA specifies the first dimension of A as declared
                                   in the calling (sub) program. When  TRANSA = 'N' or 'n' then
                                   LDA must be at least  max( 1, m ), otherwise  LDA must be at
                                   least  max( 1, k )
--ldb arg (=128)                   On entry, LDB specifies the first dimension of B as declared
                                   in the calling (sub) program. When  TRANSB = 'N' or 'n' then
                                   LDB must be at least  max( 1, k ), otherwise  LDB must be at
                                   least  max( 1, n ).
--ldc arg (=128)                   On entry, LDC specifies the first dimension of C as declared
                                   in  the  calling  (sub)  program.   LDC  must  be  at  least
                                   max( 1, m ).
--ldd arg (=128)                   On entry, LDD specifies the first dimension of D as desired
                                   in  the  calling  (sub)  program.   LDD  must  be  at  least
                                   max( 1, m ).
--alpha arg (=1)                   Specifies the scalar alpha
--beta arg (=0)                    Specifies the scalar beta
--initialization arg (=rand_int)   Intialize with random numbers, trig functions sin
                                   and cos, hpl-like input, or by loading data from 
                                   a bin file. See methods below for additional
                                   arguements required.
                                   Options: rand_int, rand_narrow, rand_broad,
                                   rand_full, trig_float, hpl, const, file
-s [ --storeInitData ] arg (=0)    Dump initialization data in to bin files? 
                                   Note: Storing is not done when loading from bin files.
                                   Please specify file names using --x_file flags 
                                   0 = No, 1 = Yes (default: No)   
-o [ --storeOutputData ] arg (=0)  Dump results matrix in to bin files? 
                                   Please specify file names using --x_file flags 
                                   0 = No, 1 = Yes (default: No)
                                   Note that multiple iterations will change results unless reinit_c flag is specified
--a_file arg                       Bin file storing matrix A.
                                   Options: text.bin 
--b_file arg                       Bin file storing matrix B.
                                   Options: text.bin 
--c_file arg                       Bin file storing matrix C.
                                   Options: text.bin 
--o_file arg                       Bin file storing result matrix.
                                   Options: text.bin 
-v [ --verify ] arg (=0)           Validate GPU results with CPU? 0 = No, 1 =
                                   Yes (default: No)
-u [ --unit_check ] arg (=0)       Unit Check? 0 = No, 1 = Yes (default: No)
-i [ --iters ] arg (=10)           Iterations to run inside timing loop
--reinit_c arg (=0)                Reinitialize C between iterations? 0 = No, 1 = Yes (default: No) 
                                   Will introduce event timer overhead. Performance with this feature 
                                   enabled is comparable to --time_each_iter==1. 
--flush_gpu_cache arg (=0)         Flush GPU L2 cache between iterations? 0 = No, 1 = Yes (default: No)
                                   Will introduce event timer overhead. Performance with this feature 
                                   enabled is comparable to --time_each_iter==1
--time_each_iter arg (=0)          Explicitly time each iteration? This introduces hipEvent overhead
                                   and is automatically enabled when reinit_c==1 or flush_gpu_cache==1  
                                   Options: 0 = No, 1 = Yes (default: No)

--tensile_timing arg (=0)          Get kernel timing from Tensile? This sends hipEvents directly to the kernel call,
                                   eliminating overhead that may be seen for smaller launches. 
                                   Will use this timing to calculate performance when enabled.  
                                   Options: 0 = No, 1 = Yes (default: No)

--device (=0)                      Set default device to be used for subsequent program runs

--multi_device (=1)                This flag is used to specify how many devices to launch work on simultaneously (default: 1)
                                   The first x amount of devices will be used (--device flag is muted). 
                                   Multiple threads will sync after setup for each device.
                                   Then a rocblas call will be deployed to each device simultaneously and the longest timing duration will be pulled.
                                   Each device will run iters iterations, and total performance will be calculated as combined iterations
                                   Flag cannot be combined with time_each_iter
```
GEMM Strided Batched requires the following additional arguments:
```
--stride_a arg (=16384)            Specific stride of strided_batched matrix A,
                                   is only applicable to strided batchedBLAS-2
                                   and BLAS-3: second dimension * leading
                                   dimension.
--stride_b arg (=16384)            Specific stride of strided_batched matrix B,
                                   is only applicable to strided batchedBLAS-2
                                   and BLAS-3: second dimension * leading
                                   dimension.
--stride_c arg (=16384)            Specific stride of strided_batched matrix C,
                                   is only applicable to strided batchedBLAS-2
                                   and BLAS-3: second dimension * leading
                                   dimension.
--batch arg (=1)                   Number of matrices. Only applicable to
                                   batched routines
```
GEMM EX requires the following arguments in addition to both of the previous lists:
```
--a_type arg (=precision)          Precision of matrix A. Options:
                                   s,d,bf16_r,f32_r,f64_r
--b_type arg (=precision)          Precision of matrix B. Options:
                                   s,d,bf16_r,f32_r,f64_r
--c_type arg (=precision)          Precision of matrix C. Options:
                                   s,d,bf16_r,f32_r,f64_r
--d_type arg (=precision)          Precision of matrix D. Options:
                                   s,d,bf16_r,f32_r,f64_r
--compute_type arg (=precision)    Precision of computation. Options:
                                   s,d,f16_r,f32_r,f64_r
--algo arg (=0)                    Extended precision gemm algorithm
--solution_index arg (=0)          Extended precision gemm solution index
--flags arg (=10)                  Extended precision gemm flags
--c_equals_d arg (=1)              Is C equal to D? 0 = No, 1 = Yes (default: Yes)
```
Note: If a precision of bf16_r is chosen, compute_type must explicitly be set to f32_r/s

Initialization Methods
---------------

This tool is designed to simulate different types of loads to test hardware for various applications. One of the following options are available to choose from:

- **Random Int**: This method intializes the input matrix A and B using randomized int values between +1 and +10. B is initilized similiarly with alternating signs. If beta is nan, matrix C is initialized with nans.
- **Random Narrow Range**: This method sets limits to the exponent bits and randomizes the sign and mantissa to intialize the input matrices with values that range from -2 to +2.
- **Random Broad Range**: This method sets limits to the exponent bits and randomizes the sign and mantissa to intialize the input matrices with a range of values that avoid overflow/underflow, and do not introduce nans.
- **Random Full Range**: This method randomizes the exponent, sign and mantissa bits to intialize the input matrices with the full range of values specified by the precision type. This is likely to introduce nans.
- **Constant**: This method uses the user input specified by the flag *--initVal* to fill the input matrices A, B and C.
- **Trig**: This method initializes the input matrices using trigonometric functions based on the index. The matrices A and C utilize the sin function, while B uses cos. 
- **HPL**: This method iniatializes the input matrices with values between -0.5 and +0.5
- **Bin file**: A user may choose to load initialization data from a bin file. The file names must be specified via the flags *--a_file*, *--b_file* and *--c_file*. This method will fail if there is not sufficient data found in the bin files with respect to the GEMM parameters; M, N and K.
Note: File are loaded from and stored to files using little endian convention.
