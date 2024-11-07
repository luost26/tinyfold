#ifndef FOLDING_LINEAR_H
#define FOLDING_LINEAR_H

#include "../matrix.h"

template <typename T>
void linear(matrix<T> &in, matrix<T> &weight, matrix<T> &bias, matrix<T> &out) {
    // in: (batch, in_channels)
    // weight: (out_channels, in_channels)
    // bias: (out_channels, 1)
    // out: (batch, out_channels)
    matmul_add<true>(in, weight, bias, out);
}

template <typename T>
void linear(matrix<T> &in, matrix<T> &weight, matrix<T> &out) {
    matmul<true>(weight, in, out);
}


#endif // FOLDING_LINEAR_H