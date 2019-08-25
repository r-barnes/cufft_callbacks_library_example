#pragma once

#include <cufft.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <iostream>

typedef cuFloatComplex cfloat;

#define CheckCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline void gpuAssert(cufftResult code, const char *file, int line, bool abort=true){
   if (code != CUFFT_SUCCESS){
      fprintf(stderr,"GPUassert: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}

namespace kernels {

void scaled_ifft2_inplace(cfloat *const data_dev_ptr, cfloat *const scaling_dev_ptr, const int n);

}