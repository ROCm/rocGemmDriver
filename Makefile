BLIS = -fopenmp -lblis -L extern/blis/lib/ -rpath extern/blis/lib/
FLAME = -lflame -L extern/flame/lib/ -rpath extern/flame/lib/
BOOST = -lboost_program_options

ifeq ($(ROCBLASPATH),)
ROCBLASLIB = -lrocblas -L /opt/rocm/rocblas/lib/
ROCBLASINCL = -I /opt/rocm/rocblas/include/
else
ifeq ($(DEBUG),1)
ROCBLASLIB = -lrocblas -L $(ROCBLASPATH)/build/debug/library/src/ -rpath $(ROCBLASPATH)/build/debug/library/src/
ROCBLASINCL = -I $(ROCBLASPATH)/library/include/ -I $(ROCBLASPATH)/build/debug/include/
else
ROCBLASLIB = -lrocblas -L $(ROCBLASPATH)/build/release/library/src/ -rpath $(ROCBLASPATH)/build/release/library/src/
ROCBLASINCL = -I $(ROCBLASPATH)/library/include/ -I $(ROCBLASPATH)/build/release/include/
endif
endif

ifeq ($(CLANG),1)
LINKER_FLAG = -rtlib=compiler-rt
endif

ifeq ($(VALIDATE),1)
ifeq ($(DEBUG),1)
GemmDriver: GemmDriver.o blis_interface.o
	/opt/rocm/hip/bin/hipcc -rtlib=compiler-rt -g -o GemmDriver GemmDriver.o blis_interface.o $(LINKER_FLAG) $(BLIS) $(FLAME)  $(ROCBLASLIB) $(BOOST)
else
GemmDriver: GemmDriver.o blis_interface.o
	/opt/rocm/hip/bin/hipcc -rtlib=compiler-rt -o GemmDriver GemmDriver.o blis_interface.o $(LINKER_FLAG) $(BLIS) $(FLAME)  $(ROCBLASLIB) $(BOOST)
endif
else
ifeq ($(DEBUG),1)
GemmDriver: GemmDriver.o
	/opt/rocm/hip/bin/hipcc -rtlib=compiler-rt -g -o GemmDriver GemmDriver.o $(LINKER_FLAG) $(ROCBLASLIB) $(BOOST)
else
GemmDriver: GemmDriver.o
	/opt/rocm/hip/bin/hipcc -rtlib=compiler-rt -o GemmDriver GemmDriver.o $(LINKER_FLAG) $(ROCBLASLIB) $(BOOST)
endif
endif

ifeq ($(VALIDATE),1)
ifeq ($(DEBUG),1)
GemmDriver.o: GemmDriver.cpp utility.hpp validate.hpp flame_interface.hpp
	/opt/rocm/hip/bin/hipcc -g -c GemmDriver.cpp -std=c++14 $(ROCBLASINCL) -I extern/flame/include/ -DVALIDATE
else
GemmDriver.o: GemmDriver.cpp utility.hpp validate.hpp flame_interface.hpp
	/opt/rocm/hip/bin/hipcc -c GemmDriver.cpp -std=c++14 $(ROCBLASINCL) -I extern/flame/include/ -DVALIDATE
endif
else
ifeq ($(DEBUG),1)
GemmDriver.o: GemmDriver.cpp utility.hpp
	/opt/rocm/hip/bin/hipcc -g -c GemmDriver.cpp -std=c++14 $(ROCBLASINCL) 
else
GemmDriver.o: GemmDriver.cpp utility.hpp
	/opt/rocm/hip/bin/hipcc -c GemmDriver.cpp -std=c++14 $(ROCBLASINCL) 
endif
endif

ifeq ($(VALIDATE),1)
ifeq ($(DEBUG),1)
blis_interface.o: blis_interface.cpp blis_interface.hpp utility.hpp
	/opt/rocm/hip/bin/hipcc -g -c blis_interface.cpp -std=c++14 -I extern/blis/include/blis/ $(ROCBLASINCL)
else
blis_interface.o: blis_interface.cpp blis_interface.hpp utility.hpp
	/opt/rocm/hip/bin/hipcc -c blis_interface.cpp -std=c++14 -I extern/blis/include/blis/ $(ROCBLASINCL)
endif
endif

clean:
	rm -f GemmDriver *.o
