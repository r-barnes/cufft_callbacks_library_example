all:
	#ordinary rdc compilation of CUDA source
	nvcc -g -G -ccbin g++ -m64 -arch=sm_61 -Xcompiler -fPIC,-g -dc -o kernels.o -c kernels.cu
	#separate device link step, necessary for rdc flow.
	#DON'T USE THIS LINE OR EVERYTHING WILL EXPLODE AND YOU WILL BE SAD.
	#nvcc -g -G -ccbin g++ -m64 -arch=sm_61 -dlink -o kernels.dlink.o kernels.o
	#creation of library - note we need ordinary linkable object and device-link object!
	nvcc -g -G -ccbin g++ -m64 -arch=sm_61 -Xcompiler -fPIC,-g -lib -o kernels.a kernels.o #kernels.dlink.o
	#host code compilation
	g++ -g -m64 -fPIC -o libmain.o -c libmain.cpp
	#host (final) link phase - the order of entries on this line is important!!
	nvcc -g -G -m64 libmain.o kernels.a -o libmain.so -shared -L/usr/local/cuda/lib64 -lcudart -lcufft_static -lculibos
	#Okay, shared library is created. Let's use it.
	g++ -o test.exe test.cpp -L. -lmain

clean:
	rm -rf *.o *.so *.exe *.a
