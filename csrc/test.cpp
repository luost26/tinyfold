#define IS_TESTING 1
#include <iostream>
#include <map>
#include <vector>
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

int main(int argc, char **argv) {
    std::map<std::string, void (*)()> tests;
    #define ADD_TEST(test) tests[#test] = test
    ADD_TEST(test_ipa);
    ADD_TEST(test_affine_quat_conversion);
    ADD_TEST(test_structure_module);
    ADD_TEST(test_adaptor);
    ADD_TEST(test_folding);
    ADD_TEST(test_transformer);
    ADD_TEST(test_esm_small);
    ADD_TEST(test_esm_full_3B);
    ADD_TEST(benchmark_transformer);
    ADD_TEST(test_pseudo_quantize);
    ADD_TEST(test_quantize_4bit);
    ADD_TEST(test_output_linear_residual_W4A32);

    bool test_all = argc == 1;
    if (test_all) {
        for (auto &test : tests) {
            test.second();
        }
        return 0;
    } else {
        for (int i = 1; i < argc; i++) {
            std::string test_name = argv[i];
            if (tests.find(test_name) == tests.end()) {
                std::cerr << "Test not found: " << test_name << std::endl;
            }
            tests[test_name]();
        }
    }
    return 0;
}