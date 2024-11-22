#include <iostream>
#include "folding/ipa_test.h"
#include "folding/structure_module_test.h"
#include "folding/adaptor_test.h"
#include "folding/test.h"
#include "plm/transformer_test.h"
#include "plm/transformer_kernels_test.h"
#include "plm/esm_test.h"
#include "tinyfold.h"
#include "matrix_q.h"
#include "matrix_q_test.h"
#include "utils/pseudo_quant.h"
#include "utils/pseudo_quant_test.h"

int main() {
    test_ipa();
    test_affine_quat_conversion();
    test_structure_module();
    test_adaptor();
    test_folding();
    test_transformer();
    test_esm_small();
    test_esm_full_3B();
    benchmark_transformer();
    test_pseudo_quantize();
    test_quantize_4bit();
    test_output_linear_residual_W4A32();
    return 0;
}