all:
	#ordinary rdc compilation of CUDA source
	nvcc -g -G -ccbin g++ -m64 -arch=sm_61 -dc -o kernels.o -c kernels.cu
	#separate device link step, necessary for rdc flow.
	#DON'T USE THIS LINE OR EVERYTHING WILL EXPLODE AND YOU WILL BE SAD.
	#nvcc -g -G -ccbin g++ -m64 -arch=sm_61 -dlink -o kernels.dlink.o kernels.o
	#creation of library - note we need ordinary linkable object and device-link object!
	nvcc -g -G -ccbin g++ -m64 -arch=sm_61 -lib -o kernels.a kernels.o #kernels.dlink.o
	#host code compilation
	g++ -g -m64 -o test.o -c test.cpp
	#host (final) link phase - the order of entries on this line is important!!
	nvcc -g -G -m64 test.o kernels.a -o test.exe -lcufft_static -lculibos

clean:
	rm -rf *.o *.so
