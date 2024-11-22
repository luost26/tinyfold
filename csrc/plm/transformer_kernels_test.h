#ifndef PLM_TRANSFORMER_KERNALS_TEST_H
#define PLM_TRANSFORMER_KERNALS_TEST_H

#include <memory>
#include "transformer_kernels.h"
#include "../utils/timer.h"
#include "../utils/pseudo_quant.h"

void test_output_linear_residual_W4A32() {
    START_TIMER();
    matrix<float> x(1280, 2560);
    for (int i = 0; i < x.numel(); i ++) {
        x.data[i] = (float)rand() / RAND_MAX;
    }

    matrix<float> w(256, 2560);
    for (int i = 0; i < w.numel(); i ++) {
        w.data[i] = (float)rand() / RAND_MAX;
    }
    matrix<float> b(256, 1);
    for (int i = 0; i < b.numel(); i ++) {
        b.data[i] = (float)rand() / RAND_MAX;
    }
    RECORD_TIME("init");

    std::unique_ptr<quantized_matrix<Q4, 128>> qw(quantize<Q4, 128>(w));
    RECORD_TIME("quantize w");
    
    std::unique_ptr<quantized_matrix<Q8, 128>> qx(quantize<Q8, 128>(x));
    RECORD_TIME("quantize x");

    pseudo_quantize_(w, 128, 4);
    RECORD_TIME("pseudo quantize x");

    matrix<float> out1(1280, 256);
    output_linear_residual(x, w, b, out1);
    RECORD_TIME("fp32");

    matrix<float> out2(1280, 256);
    output_linear_residual(x, *qw, b, out2);
    RECORD_TIME("W4A32");

    TEST_ALLCLOSE("out1", out1, out2, 1e-6f);
}

#endif // PLM_TRANSFORMER_KERNALS_TEST_H
