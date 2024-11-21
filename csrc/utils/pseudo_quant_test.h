#ifndef UTILS_PSEUDO_QUANT_TEST_H
#define UTILS_PSEUDO_QUANT_TEST_H

#include "pseudo_quant.h"

void test_pseudo_quantize() {
    std::cout << "====== Test Pseudo Quantization ======" << std::endl;

    matrix<float> mat(128, 256);
    load_(mat, "../data/c_test/pseudo_quantize/input.bin");

    pseudo_quantize_(mat, 128, 4);
    matrix<float> ref(128, 256);
    load_(ref, "../data/c_test/pseudo_quantize/quantized.bin");
    TEST_ALLCLOSE("pseudo_quantize", mat, ref, 1e-6f);
}

#endif  // UTILS_PSEUDO_QUANT_TEST_H