#ifndef FOLDING_LINEAR_H
#define FOLDING_LINEAR_H

#include "../matrix.h"

template <typename T>
void linear(matrix<T> &in, matrix<T> &weight, matrix<T> &bias, matrix<T> &out) {
    // in: (in_channels, 1)
    // weight: (out_channels, in_channels)
    // bias: (out_channels, 1)
    // out: (out_channels, 1)
    matmul_add(weight, in, bias, out);
}

template <typename T>
void linear(matrix<T> &in, matrix<T> &weight, matrix<T> &out) {
    matmul(weight, in, out);
}


#endif // FOLDING_LINEAR_H