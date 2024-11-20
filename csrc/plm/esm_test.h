#ifndef PLM_ESM_TEST_H
#define PLM_ESM_TEST_H

#include "esm.h"

void test_esm_small() {
    std::cout << "====== Test ESM (Small) ======" << std::endl;
    int seqlen = 17;
    const std::string dirpath = "../data/c_test/esm_small";
    std::cout << "seqlen = " << seqlen << std::endl;
    std::cout << "dirpath = " << dirpath << std::endl;

    std::unique_ptr<ESM> esm(load_esm(dirpath));
    matrix<float> esm_aatype_f32 = matrix<float>(dirpath + "/input/tokens.bin", seqlen, 1);
    matrix<int> esm_aatype = matrix<int>(seqlen, 1);
    for (int i = 0; i < seqlen; i ++) {
        *esm_aatype(i, 0) = (int)*esm_aatype_f32(i, 0);
    }
    std::unique_ptr<ESMBuffer> buffer(esm->create_buffer(esm_aatype));

    matrix<float> repr_wgt(esm->cfg.num_layers + 1, 1);
    *repr_wgt(esm->cfg.num_layers, 0) = 1.0f;
    ESMRepresentation repr_out(seqlen, esm->cfg, repr_wgt);

    #define TEST_INTERMEDIATE_ALLCLOSE(NAME,OUTNAME,ATOL) { \
        auto OUTNAME##_ref = matrix<float>(dirpath + "/output/" #OUTNAME ".bin", NAME.n_rows, NAME.n_cols); \
        TEST_ALLCLOSE(#OUTNAME, NAME, OUTNAME##_ref, ATOL); \
    }
    (*esm)(esm_aatype, *buffer, nullptr, 0);
    TEST_INTERMEDIATE_ALLCLOSE(buffer->x, representations_0, 1e-4f);

    (*esm)(esm_aatype, *buffer, nullptr, 1);
    TEST_INTERMEDIATE_ALLCLOSE(buffer->y, representations_1, 1e-4f);

    (*esm)(esm_aatype, *buffer, nullptr, 2);
    TEST_INTERMEDIATE_ALLCLOSE(buffer->x, representations_2, 1e-4f);

    (*esm)(esm_aatype, *buffer, nullptr, 3);
    TEST_INTERMEDIATE_ALLCLOSE(buffer->y, representations_3, 1e-4f);

    (*esm)(esm_aatype, *buffer, &repr_out, 4);
    TEST_INTERMEDIATE_ALLCLOSE(buffer->x, representations_4, 1e-4f);

    TEST_INTERMEDIATE_ALLCLOSE(repr_out.z, attentions_seq, 1e-4f);
    #undef TEST_INTERMEDIATE_ALLCLOSE

}

#endif // PLM_ESM_TEST_H