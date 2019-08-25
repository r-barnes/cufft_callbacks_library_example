#pragma once

#include <complex>
#include <vector>

void scaled_ifft_inplace(std::vector<std::complex<float>> &vec, const std::vector<std::complex<float>> &scale, const int side_size);