#ifndef MATRIX_Q_H
#define MATRIX_Q_H

#include <iostream>
#include "matrix.h"

enum NumBits {
    Q4 = 4,
    Q8 = 8,
};

template <NumBits num_bits>
struct quantized_matrix {
    unsigned char *data;
    int block_size;
    float *scales;
    int *zero_points;
    int n_rows;
    int n_cols;

    quantized_matrix(int n_rows, int n_cols, int block_size):
        n_rows(n_rows), n_cols(n_cols), block_size(block_size)
    {
        int numel = n_rows * n_cols;
        if (numel % block_size != 0) {
            std::cerr << "Matrix size is not a multiple of block size" << std::endl;
            return;
        }
        if (block_size % 8 != 0) {
            std::cerr << "Block size is not a multiple of 8" << std::endl;
            return;
        }
        int num_blocks = numel / block_size;
        scales = new float[num_blocks];
        memset(scales, 0, num_blocks * sizeof(float));
        zero_points = new int[num_blocks];
        memset(zero_points, 0, num_blocks * sizeof(int));
        if constexpr (num_bits == Q4) {
            data = new unsigned char[numel / 2];
            memset(data, 0, numel / 2);
        } else if constexpr (num_bits == Q8) {
            data = new unsigned char[numel];
            memset(data, 0, numel);
        }
    }

    ~quantized_matrix() {
        delete[] data;
        delete[] scales;
        delete[] zero_points;
    }

    int numel() const {
        return n_rows * n_cols;
    }

    inline int group_index(int row, int col) const {
        return (row * n_cols + col) / block_size;
    }

    inline int quantized_value_at(int row, int col) const {
        if constexpr (num_bits == Q8) {
            return data[row * n_cols + col];
        } else if constexpr (num_bits == Q4) {
            int elem_idx = row * n_cols + col;
            return (elem_idx % 2 == 0) ? (data[elem_idx / 2] >> 4) : (data[elem_idx / 2] & 0x0F);
        }
    }

    inline float dequantize(int row, int col) const {
        if constexpr (num_bits == Q8) {
            int w_q = data[row * n_cols + col];
            int grp_idx = group_index(row, col);
            return scales[grp_idx] * (w_q - zero_points[grp_idx]);
        } else if constexpr (num_bits == Q4) {
            int elem_idx = row * n_cols + col;
            int w_q = (elem_idx % 2 == 0) ? (data[elem_idx / 2] >> 4) : (data[elem_idx / 2] & 0x0F);
            int grp_idx = group_index(row, col);
            return scales[grp_idx] * (w_q - zero_points[grp_idx]);
        }
    }
};

template <NumBits num_bits>
quantized_matrix<num_bits>* quantize(const matrix<float> &A, int block_size, quantized_matrix<num_bits> *out = nullptr) {
    const int numel = A.numel();
    if (numel % block_size != 0) {
        std::cerr << "Matrix size is not a multiple of block size" << std::endl;
        return nullptr;
    }
    const int num_blocks = A.numel() / block_size;
    
    constexpr int max_int = (1 << (int)num_bits) - 1;
    quantized_matrix<num_bits> *q_matrix = out;

    if (q_matrix == nullptr) {
        q_matrix = new quantized_matrix<num_bits>(A.n_rows, A.n_cols, block_size);
    } else {
        if (q_matrix->n_rows != A.n_rows || q_matrix->n_cols != A.n_cols || q_matrix->block_size != block_size) {
            std::cerr << "Output matrix size mismatch" << std::endl;
            exit(1);
        }
    }

    #pragma omp parallel for
    for (int block_idx = 0; block_idx < num_blocks; block_idx ++) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::min();
        for (int i = 0; i < block_size; i ++) {
            float val = A.data[block_idx * block_size + i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        float scale = std::max(max_val - min_val, 1e-5f) / max_int;
        int zero_point = std::min(std::max(-(int)std::round(min_val / scale), 0), max_int);

        q_matrix->scales[block_idx] = scale;
        q_matrix->zero_points[block_idx] = zero_point;

        if constexpr (num_bits == Q8) {
            for (int i = 0; i < block_size; i ++) {
                float w = A.data[block_idx * block_size + i];
                int w_q = (int)std::round(w / scale) + zero_point;
                w_q = std::min(std::max(w_q, 0), max_int);
                q_matrix->data[block_idx * block_size + i] = w_q;
            }
        } else if constexpr (num_bits == Q4) {
            for (int i = 0; i < block_size; i += 2) {
                float w1 = A.data[block_idx * block_size + i];
                float w2 = A.data[block_idx * block_size + i + 1];
                int w1_q = (int)std::round(w1 / scale) + zero_point;
                w1_q = std::min(std::max(w1_q, 0), max_int);
                int w2_q = (int)std::round(w2 / scale) + zero_point;
                w2_q = std::min(std::max(w2_q, 0), max_int);
                q_matrix->data[(block_idx * block_size + i) / 2] = (w1_q << 4) | w2_q;
            }
        }
    }
    return q_matrix;
}

template <NumBits num_bits>
std::ostream& operator<<(std::ostream &os, const quantized_matrix<num_bits> &A)
{

    #define PRINT_ROW_ELEMENTS(MAT,ROW,LOW,UP) { \
        for (int j = LOW; j < UP; j++) { \
            os << MAT.dequantize(ROW, j) << "(" << MAT.quantized_value_at(ROW, j) << ")"; \
            if (j < UP - 1) os << " "; \
        } \
    }

    #define PRINT_ROW(MAT,ROW,NUM_COLS) { \
        os << "["; \
        if (NUM_COLS <= 10) { \
            PRINT_ROW_ELEMENTS(MAT,ROW,0,NUM_COLS); \
        } else { \
            PRINT_ROW_ELEMENTS(MAT,ROW,0,5); \
            os << "... "; \
            PRINT_ROW_ELEMENTS(MAT,ROW,NUM_COLS-5,NUM_COLS); \
        } \
        os << "]\n"; \
    }

    os << "[";
    if (A.n_rows <= 10) {
        for (int i = 0; i < A.n_rows; i++) {
            PRINT_ROW(A,i,A.n_cols);
        }
    } else {
        for (int i = 0; i < 5; i++) {
            PRINT_ROW(A,i,A.n_cols);
        }
        os << "...\n";
        for (int i = A.n_rows-5; i < A.n_rows; i++) {
            PRINT_ROW(A,i,A.n_cols);
        }
    }
    #undef PRINT_ROW_ELEMENTS
    #undef PRINT_ROW
    os << ", size=(" << A.n_rows << ", " << A.n_cols << ")]\n";
    return os;
}

#endif // MATRIX_Q_H
