#ifndef FOLDING_LINEAR_H
#define FOLDING_LINEAR_H

#include "../matrix.h"

template <typename T>
void linear(const matrix<T> &in, const matrix<T> &weight, const matrix<T> &bias, matrix<T> &out) {
    // in: (batch, in_channels)
    // weight: (out_channels, in_channels)
    // bias: (out_channels, 1)
    // out: (batch, out_channels)
    matmul_add<true, true>(in, weight, bias, out);
}

template <typename T>
void linear(const matrix<T> &in, const matrix<T> &weight, matrix<T> &out) {
    matmul<true>(in, weight, out);
}

template <typename T>
void linear_residual(const matrix<T> &in, const matrix<T> &weight, const matrix<T> &bias, matrix<T> &out) {
    // in: (batch, in_channels)
    // weight: (out_channels, in_channels)
    // bias: (out_channels, 1)
    // out: (batch, out_channels)
    matmul_add<true, true, true>(in, weight, bias, out);
}

template <typename T>
void linear_residual(const matrix<T> &in, const matrix<T> &weight, matrix<T> &out) {
    matmul<true, true>(in, weight, out);
}


#endif // FOLDING_LINEAR_H