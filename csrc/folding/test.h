#ifndef FOLDING_TEST_H
#define FOLDING_TEST_H

#include <iostream>
#include <memory>
#include "adaptor.h"
#include "structure_module.h"


void test_folding() {
    std::cout << "====== Test Folding (Adaptor + Structure Module) ======" << std::endl;
    int seqlen = 128;
    const std::string dirpath = "../data/c_test/esmfold_folding_only";

    std::unique_ptr<Adaptor> adaptor(load_adaptor(dirpath + "/adaptor"));
    std::unique_ptr<AdaptorBuffer> buffer(adaptor->create_buffer(seqlen));

    std::unique_ptr<StructureModule> model(load_structure_module(dirpath + "/structure_module"));
    std::unique_ptr<StructureModuleBuffer> sm_buffer(model->create_buffer(seqlen));

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

    std::cout << "Running Adaptor" << std::endl;
    (*adaptor)(esm_s, esm_z, aatype, residx, *buffer);

    #define TEST_INTERMEDIATE_ALLCLOSE(NAME,OUTNAME,ATOL) { \
        auto NAME##_ref = matrix<float>(dirpath + "/output/" #OUTNAME ".bin", buffer->NAME.n_rows, buffer->NAME.n_cols); \
        TEST_ALLCLOSE(#NAME, buffer->NAME, NAME##_ref, ATOL); \
    }
    TEST_INTERMEDIATE_ALLCLOSE(sm_s, sm_input_single, 2e-3f);
    TEST_INTERMEDIATE_ALLCLOSE(z, sm_input_pair, 2e-4f);
    #undef TEST_INTERMEDIATE_ALLCLOSE

    std::cout << "Running Structure Module" << std::endl;
    (*model)(buffer->sm_s, buffer->z, *sm_buffer);

    #define TEST_INTERMEDIATE_ALLCLOSE(NAME,OUTNAME,ATOL) { \
        auto NAME##_ref = matrix<float>(dirpath + "/output/" #OUTNAME ".bin", sm_buffer->NAME.n_rows, sm_buffer->NAME.n_cols); \
        TEST_ALLCLOSE(#NAME, sm_buffer->NAME, NAME##_ref, ATOL); \
    }
    TEST_INTERMEDIATE_ALLCLOSE(r, frames_affine, 1e-3f);
    #undef TEST_INTERMEDIATE_ALLCLOSE

    std::cout << sm_buffer->r << std::endl;
}



#endif
