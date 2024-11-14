#ifndef FOLDING_STRUCTURE_MODULE_TEST_H
#define FOLDING_STRUCTURE_MODULE_TEST_H

#include <iostream>
#include <memory>
#include "structure_module.h"

void test_structure_module() {
    std::cout << "====== Test Structure Module ======" << std::endl;
    int seqlen = 17;
    const std::string dirpath = "../data/c_test/structure_module";
    std::cout << "seqlen = " << seqlen << std::endl;
    std::cout << "dirpath = " << dirpath << std::endl;

    std::unique_ptr<StructureModule> model(load_structure_module("../data/c_test/structure_module"));
    matrix<float> s(dirpath + "/input/s.bin", seqlen, model->cfg.c_s);
    matrix<float> z(dirpath + "/input/z.bin", seqlen * seqlen, model->cfg.c_z);
    matrix<int64_t> aatype(dirpath + "/input/aatype.bin", seqlen, 4*4);
    std::unique_ptr<StructureModuleBuffer> buffer(model->create_buffer(seqlen));

    (*model)(s, z, *buffer);

    #define TEST_INTERMEDIATE_ALLCLOSE(NAME,OUTNAME,ATOL) { \
        auto NAME##_ref = matrix<float>(dirpath + "/output/" #OUTNAME ".bin", buffer->NAME.n_rows, buffer->NAME.n_cols); \
        TEST_ALLCLOSE(#NAME, buffer->NAME, NAME##_ref, ATOL); \
    }
    TEST_INTERMEDIATE_ALLCLOSE(z, z_1, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(s, s_ipa_7, 1e-6f);
    #undef TEST_INTERMEDIATE_ALLCLOSE
}

#endif // FOLDING_STRUCTURE_MODULE_TEST_H
