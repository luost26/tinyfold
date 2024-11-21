#ifndef PLM_TRANSFORMER_KERNALS_H
#define PLM_TRANSFORMER_KERNALS_H

#include "../matrix.h"
#include "../matrix_q.h"


inline float rotary_angle(int s, int d, int dim) {
    const float inv_freq = 1.0f / powf(10000.0f, (float)((d * 2) % dim) / (float)dim);
    const float angle = (float)s * inv_freq;
    return angle;
}

inline void apply_rotary_embedding_(matrix<float> &x, int num_heads) {
    const int dim = x.n_cols;
    const int seqlen = x.n_rows / num_heads;
    const int half_dim = dim / 2;
    #pragma omp parallel for
    for (int i = 0; i < num_heads * seqlen; ++i) {
        const int head_idx = i / seqlen;
        const int seq_idx = i % seqlen;
        for (int j = 0; j < half_dim; j ++) {
            float angle_lo = rotary_angle(seq_idx, j, dim);
            float c_lo = cosf(angle_lo);
            float s_lo = sinf(angle_lo);
            float x_lo = *x(i, j) * c_lo - *x(i, j + half_dim) * s_lo;

            float angle_hi = rotary_angle(seq_idx, j + half_dim, dim);
            float c_hi = cosf(angle_hi);
            float s_hi = sinf(angle_hi);
            float x_hi = *x(i, j + half_dim) * c_hi + *x(i, j) * s_hi;

            *x(i, j) = x_lo;
            *x(i, j + half_dim) = x_hi;
        }
    }
}

template <typename T>
void attn_proj_linear(const matrix<T> &in, const matrix<T> &weight, const matrix<T> &bias, matrix<T> &out) {
    // in: (seqlen, in_dim)
    // weight: (num_heads * head_dim, in_dim)
    // bias: (num_heads * head_dim, 1)
    // out: (num_heads * seqlen, head_dim)
    int seqlen = in.n_rows;
    int in_dim = in.n_cols;
    int head_dim = out.n_cols;
    int num_heads = out.n_rows / seqlen;
    #pragma omp parallel for
    for (int i = 0; i < num_heads * seqlen; i ++) {
        const int head_idx = i / seqlen;
        const int seq_idx = i % seqlen;
        for (int j = 0; j < head_dim; j ++) {
            T sum = *bias(head_idx * head_dim + j, 0);
            for (int k = 0; k < in_dim; k ++) {
                sum += *in(seq_idx, k) * *weight(head_idx * head_dim + j, k);
            }
            *out(head_idx * seqlen + seq_idx, j) = sum;
        }
    }
}

template <typename T>
void attn_out_linear(const matrix<T> &in, const matrix<T> &weight, const matrix<T> &bias, matrix<T> &out) {
    // in: (num_heads * seqlen, head_dim)
    // weight: (out_dim, num_heads * head_dim)
    // bias: (out_dim, 1)
    // out: (seqlen, out_dim)
    const int head_dim = in.n_cols;
    const int out_dim = weight.n_rows;
    const int seqlen = out.n_rows;
    const int num_heads = in.n_rows / seqlen;

    #pragma omp parallel for
    for (int i = 0; i < seqlen; i ++) {
        for (int j =0; j < out_dim; j ++) {
            T sum = *bias(j, 0);
            for (int k = 0; k < num_heads * head_dim; k ++) {
                const int head_idx = k / head_dim;
                const int dim_idx = k % head_dim;
                sum += *in(head_idx * seqlen + i, dim_idx) * *weight(j, k);
            }
            *out(i, j) = sum;
        }
    }
}

template <bool transposed_B = false, typename T>
void bmm(const matrix<T> &A, const matrix<T> &B, matrix<T> &C) {
    // A:     (bsz * N, K)
    // B(tr): (bsz * M, K)
    // B:     (bsz * K, M)
    // C:     (bsz * N, M)
    if (&A == &C || &B == &C)
    {
        std::cerr << "Matrix multiplication cannot be done inplace" << std::endl;
        exit(1);
    }

    const int M = C.n_cols;
    const int K = A.n_cols;
    int bsz = transposed_B ? B.n_rows / M : B.n_rows / K;
    const int N = A.n_rows / bsz;

    #pragma omp parallel for
    for (int i = 0; i < bsz * N; i ++) { 
        const int batch_idx = i / N;
        const int n_idx = i % N;
        for (int j = 0; j < M; j ++) {
            T sum = 0;
            for (int k = 0; k < K; k ++) {
                if constexpr (transposed_B) {
                    sum += *A(i, k) * *B(batch_idx * M + j, k);
                } else {
                    sum += *A(i, k) * *B(batch_idx * K + k, j);
                }
            }
            *C(i, j) = sum;
        }
    }
}


template <typename T>
void fused_layer_norm_linear_gelu(const matrix<T> &in, const matrix<T> &norm_weight, const matrix<T> &norm_bias, const matrix<T> &linear_weight, const matrix<T> &linear_bias, matrix<T> &out) {
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
            sum = sum * 0.5 * (1.0 + erf(sum / SQRT2));
            *out(i, j) = sum;
        }
    }
}

template <typename T1, typename T2, typename T3, typename T4>
void _check_output_linear_residual_shape(const T1 &A, const T2 &B, const T3 &bias, const T4 &C) {
    if ((void*)&A == (void*)&C || (void*)&B == (void*)&C) {
        std::cerr << "Matrix multiplication cannot be done inplace" << std::endl;
        exit(1);
    }
    int I, J, K1, K2;
    I = A.n_rows;
    J = B.n_rows;
    K1 = A.n_cols;
    K2 = B.n_cols;
    if (C.n_rows != I || C.n_cols != J || K1 != K2)
    {
        std::cerr << "Matrix dimensions do not match: ";
        std::cerr << "A(" << A.n_rows << ", " << A.n_cols << ") ";
        std::cerr << "B(" << B.n_rows << ", " << B.n_cols << ") ";
        std::cerr << "bias(" << bias.n_rows << ", " << bias.n_cols << ") ";
        std::cerr << "out(" << C.n_rows << ", " << C.n_cols << ") " << std::endl;
        exit(1);
    }
    if (bias.n_cols != 1) {
        std::cerr << "Bias vector must have 1 column" << std::endl;
        exit(1);
    }
    if (bias.n_rows != C.n_cols) {
        std::cerr << "Bias vector must have the same number of rows as the output matrix's number of cols" << std::endl;
        exit(1);
    }
}

void output_linear_residual(const matrix<float> &A, const matrix<float> &B, const matrix<float> &bias, matrix<float> &C) {
    // A: (bsz, in_channels)
    // B: (out_channels, in_channels)
    // bias: (out_channels, 1)
    // C: (bsz, out_channels)
    _check_output_linear_residual_shape(A, B, bias, C);
    #pragma omp parallel for
    for (int i = 0; i < C.n_rows; i++) {
        for (int j = 0; j < C.n_cols; j++) {
            float sum = *bias(j, 0);
            for (int k = 0; k < A.n_cols; k++) {
                sum += *A(i, k) * *B(j, k);
            }
            *C(i, j) += sum;
        }
    }
}

template <int block_size>
void output_linear_residual(const matrix<float> &A, const quantized_matrix<Q4, block_size> &B, const matrix<float> &bias, matrix<float> &C) {
    matrix<float> *B_fp32 = new matrix<float>(B.n_rows, B.n_cols);
    B.dequantize(*B_fp32);
    output_linear_residual(A, *B_fp32, bias, C);
    delete B_fp32;
}

template <int block_size>
void output_linear_residual_(const matrix<float> &A, const quantized_matrix<Q4, block_size> &B, const matrix<float> &bias, matrix<float> &C) {
    _check_output_linear_residual_shape(A, B, bias, C);
    #pragma omp parallel for
    for (int i = 0; i < C.n_rows; i++) {
        for (int j = 0; j < C.n_cols; j++) {
            float sum = *bias(j, 0);
            for (int k = 0; k < A.n_cols; k += block_size) {
                int grp_idx = B.group_index(j, k);
                float scale = B.scales[grp_idx];
                int zero_point = B.zero_points[grp_idx];

                float inner_sum = 0.0f;
                for (int l = 0; l < block_size; l += 8) {
                    int elem_idx = B.elem_index(j, k + l);
                    unsigned int wq_packed8 = ((unsigned int*)B.data)[elem_idx / 4];
                    for (int m = 0; m < 8; m++) {
                        int wq = (wq_packed8 >> (m * 4)) & 0x0F;
                        float a = *A(i, k + l + m);
                        inner_sum += a * (wq - zero_point);
                    }
                }
                sum += scale * inner_sum;
            }
            *C(i, j) += sum;
        }
    }
}

template <int block_size>
void output_linear_residual(const quantized_matrix<Q8, block_size> &A, const quantized_matrix<Q4, block_size> &B, const matrix<float> &bias, matrix<float> &C) {
    _check_output_linear_residual_shape(A, B, bias, C);
    std::cerr << "Not implemented" << std::endl;
    exit(1);
}


#endif // PLM_TRANSFORMER_KERNALS_H