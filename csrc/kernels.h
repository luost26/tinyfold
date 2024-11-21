#ifndef KERNELS_H
#define KERNELS_H

#include <omp.h>
#include "matrix.h"

template <ActivationType act_type = None, typename T>
inline void linear(const matrix<T> &in, const matrix<T> &weight, const matrix<T> &bias, matrix<T> &out) {
    // in: (batch, in_channels)
    // weight: (out_channels, in_channels)
    // bias: (out_channels, 1)
    // out: (batch, out_channels)
    matmul_add<true, true, false, act_type>(in, weight, bias, out);
}

template <ActivationType act_type = None, typename T>
inline void linear(const matrix<T> &in, const matrix<T> &weight, matrix<T> &out) {
    matmul<true, false, false, act_type>(in, weight, out);
}

template <typename T>
inline void linear_residual(const matrix<T> &in, const matrix<T> &weight, const matrix<T> &bias, matrix<T> &out) {
    // in: (batch, in_channels)
    // weight: (out_channels, in_channels)
    // bias: (out_channels, 1)
    // out: (batch, out_channels)
    matmul_add<true, true, true>(in, weight, bias, out);
}

template <typename T>
inline void linear_residual(const matrix<T> &in, const matrix<T> &weight, matrix<T> &out) {
    matmul<true, true, false>(in, weight, out);
}

template <ActivationType act_type = None, typename T>
void linear_(matrix<T> &x, const matrix<T> &weight, const matrix<T> &bias, matrix<T> * inplace_linear_buffer_ptr = nullptr) {
    // in: (batch, channels)
    // weight: (channels, channels)
    // bias: (channels, 1)
    int bsz = x.n_rows;
    int channels = x.n_cols;
    if (weight.n_rows != channels || weight.n_cols != channels) {
        std::cerr << "linear_: weight shape mismatch, and the channels of input and output must be the same" << std::endl;
        exit(1);
    }

    bool buffer_owned = false;
    T * buffer_all = nullptr;
    if (inplace_linear_buffer_ptr == nullptr) {
        int num_threads = omp_get_max_threads();
        std::cerr << "num_threads: " << num_threads << std::endl;
        buffer_all = new T[num_threads * channels];
        buffer_owned = true;
    } else {
        buffer_all = inplace_linear_buffer_ptr->data;
    }

    #pragma omp parallel for
    for (int i = 0; i < bsz; i ++) {
        int thread_idx = omp_get_thread_num();
        T * buffer = buffer_all + thread_idx * channels;
        for (int j = 0; j < channels; j ++) {
            T sum = *bias(j, 0);
            for (int k = 0; k < channels; k ++) {
                sum += *x(i, k) * *weight(j, k);
            }
            if (act_type == ReLU) {
                sum = sum > 0 ? sum : 0;
            }
            buffer[j] = sum;
        }
        for (int j = 0; j < channels; j ++) {
            *x(i, j) = buffer[j];
        }
    }

    if (buffer_owned) {
        delete buffer_all;
    }
}

template <typename T>
void layer_norm(const matrix<T> &in, const matrix<T> &weight, const matrix<T> &bias, matrix<T> &out) {
    // in: (batch, in_channels)
    // weight: (in_channels, 1)
    // bias: (in_channels, 1)
    // out: (batch, in_channels)
    
    #pragma omp parallel for
    for (int i = 0; i < in.n_rows; i ++) {
        T sum = 0;
        for (int j = 0; j < in.n_cols; j ++) {
            sum += *in(i, j);
        }
        T mean = sum / in.n_cols;
        T var = 0;
        for (int j = 0; j < in.n_cols; j ++) {
            T diff = *in(i, j) - mean;
            var += diff * diff;
        }
        var /= in.n_cols;
        T inv_std = 1.0f / sqrt(var + 1e-5f);
        for (int j = 0; j < in.n_cols; j ++) {
            *out(i, j) = (*in(i, j) - mean) * inv_std * *weight(j, 0) + *bias(j, 0);
        }
    }
}

template <ActivationType act_type = None, typename T>
void fused_layer_norm_linear(const matrix<T> &in, const matrix<T> &norm_weight, const matrix<T> &norm_bias, const matrix<T> &linear_weight, const matrix<T> &linear_bias, matrix<T> &out) {
    // in: (batch, in_channels)
    // weight: (out_channels, in_channels)
    // norm_weight: (in_channels, 1)
    // norm_bias: (in_channels, 1)
    // linear_bias: (out_channels, 1)
    // out: (batch, out_channels)
    int bsz = in.n_rows;
    int in_channels = in.n_cols;
    int out_channels = out.n_cols;

    #pragma omp parallel for
    for (int i = 0; i < bsz; i ++) {
        T mean = 0;
        for (int j = 0; j < in_channels; j ++) {
            mean += *in(i, j);
        }
        mean /= in_channels;

        T var = 0;
        for (int j = 0; j < in_channels; j ++) {
            T diff = *in(i, j) - mean;
            var += diff * diff;
        }
        var /= in_channels;
        T inv_std = 1.0f / sqrt(var + 1e-5f);

        for (int j = 0; j < out_channels; j ++) {
            T sum = *linear_bias(j, 0);
            for (int k = 0; k < in_channels; k ++) {
                auto in_0 = *in(i, k);
                auto nw_0 = *norm_weight(k, 0);
                auto nb_0 = *norm_bias(k, 0);
                auto lw_0 = *linear_weight(j, k);
                sum += ((in_0 - mean) * inv_std * nw_0 + nb_0) * lw_0;
            }
            if constexpr (act_type == ReLU) {
                sum = sum > 0 ? sum : 0;
            } else if (act_type == GELU) {
                sum = sum * 0.5 * (1.0 + erf(sum / SQRT2));
            }
            *out(i, j) = sum;
        }
    }
    
}

#endif // KERNELS_H