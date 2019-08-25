#include <cassert>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include <stdio.h>

#include "kernels.h"

namespace kernels {

__device__ cufftComplex cu_ifft_prescale(
    void *data_ptr,
    size_t offset,
    void *callerInfo,
    void *sharedPtr)
{
  cfloat element = ((cfloat*)data_ptr)[offset];
  cfloat tf_element = ((cfloat*)callerInfo)[offset];
  return cuCmulf(element,tf_element);
}



__device__ void cu_ifft_post_normalize(
  void *data_ptr,
  size_t offset,
  cufftComplex element,
  void *callerInfo,
  void *sharedPtr
){
  element.x /= (8*8);
  element.y /= (8*8);
  ((cfloat*)data_ptr)[offset] = element;
}



__device__ cufftCallbackLoadC  d_loadCallbackPtr  = cu_ifft_prescale;
__device__ cufftCallbackStoreC d_storeCallbackPtr = cu_ifft_post_normalize;

void scaled_ifft2_inplace(cfloat *const data_dev_ptr, cfloat *const scaling_dev_ptr, const int n){
  cufftHandle fft_plan;
  cufftPlan2d(&fft_plan, n, n, CUFFT_C2C);

  cufftCallbackLoadC h_loadCallbackPtr;
  cufftCallbackStoreC h_storeCallbackPtr;
  CheckCudaErrors(cudaMemcpyFromSymbol(&h_loadCallbackPtr,
                                        d_loadCallbackPtr,
                                        sizeof(h_loadCallbackPtr)));
  CheckCudaErrors(cudaMemcpyFromSymbol(&h_storeCallbackPtr,
                                        d_storeCallbackPtr,
                                        sizeof(h_storeCallbackPtr)));

  CheckCudaErrors(cufftXtSetCallback(fft_plan,
                          (void **)&h_loadCallbackPtr,
                          CUFFT_CB_LD_COMPLEX,
                          (void**)&scaling_dev_ptr));

  CheckCudaErrors(cufftXtSetCallback(fft_plan,
                              (void **)&h_storeCallbackPtr,
                              CUFFT_CB_ST_COMPLEX,
                              0));

  CheckCudaErrors(cudaDeviceSynchronize());

  CheckCudaErrors(cufftExecC2C(fft_plan, data_dev_ptr, data_dev_ptr, CUFFT_INVERSE));

  CheckCudaErrors(cudaDeviceSynchronize());
  CheckCudaErrors(cufftDestroy(fft_plan));
}

}