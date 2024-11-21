#ifndef MATRIX_Q_TEST_H
#define MATRIX_Q_TEST_H

#include "matrix_q.h"
#include "utils/pseudo_quant.h"

void test_quantize_8bit() {
    matrix<float> A(128, 256);
    // Fill A with random values
    for (int i = 0; i < A.numel(); i ++) {
        A.data[i] = (float)rand() / RAND_MAX;
    }
    // std::cerr << "Original: " << A << std::endl;

    auto *Q = quantize<Q8>(A, 128);
    pseudo_quantize_(A, 128, 8);

    // std::cerr << "A: " << A << std::endl;
    // std::cerr << "Q: " << *Q << std::endl;

    float max_error = 0.0f;
    for (int i = 0; i < A.n_rows; i ++) {
        for (int j = 0; j < A.n_cols; j ++) {
            float error = std::abs(*A(i, j) - Q->dequantize(i, j));
            max_error = std::max(max_error, error);
        }
    }
    std::cerr << "Max error: " << max_error << std::endl;
    if (max_error < 1e-5) {
        std::cerr << "Test passed" << std::endl;
    } else {
        std::cerr << "Test failed" << std::endl;
    }
    delete Q;
}

void test_quantize_4bit() {
    matrix<float> A(128, 256);
    // Fill A with random values
    for (int i = 0; i < A.numel(); i ++) {
        A.data[i] = (float)rand() / RAND_MAX;
    }
    // std::cerr << "Original: " << A << std::endl;

    auto *Q = quantize<Q4>(A, 128);
    pseudo_quantize_(A, 128, 4);

    // std::cerr << "A: " << A << std::endl;
    // std::cerr << "Q: " << *Q << std::endl;

    float max_error = 0.0f;
    for (int i = 0; i < A.n_rows; i ++) {
        for (int j = 0; j < A.n_cols; j ++) {
            float error = std::abs(*A(i, j) - Q->dequantize(i, j));
            max_error = std::max(max_error, error);
        }
    }
    std::cerr << "Max error: " << max_error << std::endl;
    if (max_error < 1e-5) {
        std::cerr << "Test passed" << std::endl;
    } else {
        std::cerr << "Test failed" << std::endl;
    }
    delete Q;
}

#endif // MATRIX_Q_TEST_H