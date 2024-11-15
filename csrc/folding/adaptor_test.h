#ifndef FOLDING_ADAPTOR_TEST_H
#define FOLDING_ADAPTOR_TEST_H

#include <iostream>
#include <memory>
#include "adaptor.h"


void test_adaptor() {
    std::cout << "====== Test Adaptor ======" << std::endl;
    int seqlen = 128;
    const std::string dirpath = "../data/c_test/esmfold_folding_only";

    std::unique_ptr<Adaptor> adaptor(load_adaptor(dirpath + "/adaptor"));

    std::cout << "Loading ESM features and attention matrices" << std::endl;
    matrix<float> esm_s(dirpath + "/input/esm_s.bin", seqlen, adaptor->cfg.esm_feats);
    matrix<float> esm_z(dirpath + "/input/esm_z.bin", seqlen * seqlen, adaptor->cfg.esm_attns);
    matrix<float> aatype_f(dirpath + "/input/aatype.bin", seqlen, 1);
    matrix<float> residx_f(dirpath + "/input/residx.bin", seqlen, 1);

    matrix<int> aatype(seqlen, 1);
    matrix<int> residx(seqlen, 1);
    for (int i = 0; i < seqlen; i++) {
        *aatype(i, 0) = std::round(*aatype_f(i, 0));
        *residx(i, 0) = std::round(*residx_f(i, 0));
    }

    std::unique_ptr<AdaptorBuffer> buffer(adaptor->create_buffer(seqlen));
    std::cout << "Running Adaptor" << std::endl;
    (*adaptor)(esm_s, esm_z, aatype, residx, *buffer);

    #define TEST_INTERMEDIATE_ALLCLOSE(NAME,OUTNAME,ATOL) { \
        auto NAME##_ref = matrix<float>(dirpath + "/output/" #OUTNAME ".bin", buffer->NAME.n_rows, buffer->NAME.n_cols); \
        TEST_ALLCLOSE(#NAME, buffer->NAME, NAME##_ref, ATOL); \
    }
    TEST_INTERMEDIATE_ALLCLOSE(sm_s, sm_input_single, 2e-3f);
    TEST_INTERMEDIATE_ALLCLOSE(z, sm_input_pair, 2e-4f);
    #undef TEST_INTERMEDIATE_ALLCLOSE
}


#endif // FOLDING_ADAPTOR_TEST_H
