#include <cassert>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include <stdio.h>

#include "kernels.h"

namespace kernels {

cufftHandle fft_plan = 0;
void create_fft_plan(const int side_size){
  CheckCudaErrors(cufftPlan2d(&fft_plan, side_size, side_size, CUFFT_C2C));
  CheckCudaErrors(cudaDeviceSynchronize());
}



void destroy_fft_plan(){
  CheckCudaErrors(cufftDestroy(fft_plan));
}



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

void scaled_ifft2_inplace(cfloat *const data_dev_ptr, cfloat *const scaling_dev_ptr, const int n, const bool generate_plan){
  cufftHandle my_fft_plan;

  if(generate_plan){
    cufftPlan2d(&my_fft_plan, n, n, CUFFT_C2C);
  } else if(!fft_plan){
    throw std::runtime_error("No FFT plan!");
  } else {
    my_fft_plan = fft_plan;
  }

  cufftCallbackLoadC h_loadCallbackPtr;
  cufftCallbackStoreC h_storeCallbackPtr;
  CheckCudaErrors(cudaMemcpyFromSymbol(&h_loadCallbackPtr,
                                        d_loadCallbackPtr,
                                        sizeof(h_loadCallbackPtr)));
  CheckCudaErrors(cudaMemcpyFromSymbol(&h_storeCallbackPtr,
                                        d_storeCallbackPtr,
                                        sizeof(h_storeCallbackPtr)));

  CheckCudaErrors(cufftXtSetCallback(my_fft_plan,
                          (void **)&h_loadCallbackPtr,
                          CUFFT_CB_LD_COMPLEX,
                          (void**)&scaling_dev_ptr));

  CheckCudaErrors(cufftXtSetCallback(my_fft_plan,
                              (void **)&h_storeCallbackPtr,
                              CUFFT_CB_ST_COMPLEX,
                              0));

  CheckCudaErrors(cudaDeviceSynchronize());

  CheckCudaErrors(cufftExecC2C(my_fft_plan, data_dev_ptr, data_dev_ptr, CUFFT_INVERSE));

  CheckCudaErrors(cudaDeviceSynchronize());

  if(generate_plan){
    CheckCudaErrors(cufftDestroy(my_fft_plan));
  }
}

}