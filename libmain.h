#pragma once

#include <complex>
#include <vector>

void create_fft_plan(const int side_size);
void destroy_fft_plan();
void scaled_ifft_inplace(std::vector<std::complex<float>> &vec, const std::vector<std::complex<float>> &scale, const int side_size);