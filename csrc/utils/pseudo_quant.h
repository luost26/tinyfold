#ifndef UTILS_PSEUDO_QUANT_H
#define UTILS_PSEUDO_QUANT_H

#include "../matrix.h"

void pseudo_quantize(const matrix<float> &A, int block_size, int num_bits, matrix<float> &out) {
    int numel = A.numel();
    if (numel % block_size != 0) {
        std::cerr << "pseudo_quantize_: number of elements must be divisible by block_size" << std::endl;
        exit(1);
    }
    int num_blocks = numel / block_size;
    for (int block_idx = 0; block_idx < num_blocks; block_idx ++) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::min();
        for (int i = 0; i < block_size; i ++) {
            float val = A.data[block_idx * block_size + i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        int max_int = (1 << num_bits) - 1;
        float scale = std::max(max_val - min_val, 1e-5f) / max_int;
        int zero_point = std::min(std::max(-(int)std::round(min_val / scale), 0), max_int);

        for (int i = 0; i < block_size; i ++) {
            float w = A.data[block_idx * block_size + i];
            int w_q = (int)std::round(w / scale) + zero_point;
            w_q = std::min(std::max(w_q, 0), max_int);
            float w_dq = scale * (w_q - zero_point);
            out.data[block_idx * block_size + i] = w_dq;
        }
    }
}

void pseudo_quantize_(matrix<float> &A, int block_size, int num_bits) {
    pseudo_quantize(A, block_size, num_bits, A);
}

#endif  // UTILS_PSEUDO_QUANT_H