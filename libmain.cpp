#include <cuda_runtime_api.h>

#include "kernels.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

void scaled_ifft_inplace(std::vector<std::complex<float>> &vec, const std::vector<std::complex<float>> &scale, const int side_size){
    const auto bytes_len = side_size*side_size*sizeof(cfloat);

    cfloat *dvec;
    CheckCudaErrors(cudaMalloc((void **)&dvec, bytes_len));
    CheckCudaErrors(cudaMemcpy(dvec, vec.data(), bytes_len, cudaMemcpyHostToDevice));

    cfloat *dscale;
    CheckCudaErrors(cudaMalloc((void **)&dscale, bytes_len));
    CheckCudaErrors(cudaMemcpy(dscale, scale.data(), bytes_len, cudaMemcpyHostToDevice));

    kernels::scaled_ifft2_inplace(dvec, dscale, side_size);

    CheckCudaErrors(cudaMemcpy(vec.data(), dvec, bytes_len, cudaMemcpyDeviceToHost));
}
