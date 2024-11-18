#ifndef PLM_TRANSFORMER_TEST_H
#define PLM_TRANSFORMER_TEST_H

#include <iostream>
#include <memory>
#include "transformer.h"

void test_transformer() {
    std::cout << "====== Test TransformerLayer ======" << std::endl;
    int seqlen = 17;
    const std::string dirpath = "../data/c_test/transformer";
    std::cout << "seqlen = " << seqlen << std::endl;
    std::cout << "dirpath = " << dirpath << std::endl;

    std::unique_ptr<TransformerLayer> transformer(load_transformer_layer(dirpath));
    matrix<float> x = matrix<float>(dirpath + "/input/x.bin", seqlen, transformer->cfg.embed_dim);
    std::unique_ptr<TransformerBuffer> buffer(transformer->create_buffer(seqlen));

    (*transformer)(x, *buffer);

    #define TEST_INTERMEDIATE_ALLCLOSE(NAME,OUTNAME,ATOL) { \
        auto NAME##_ref = matrix<float>(dirpath + "/output/" #OUTNAME ".bin", buffer->NAME.n_rows, buffer->NAME.n_cols); \
        TEST_ALLCLOSE(#NAME, buffer->NAME, NAME##_ref, ATOL); \
    }
    TEST_INTERMEDIATE_ALLCLOSE(q, q_rot, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(k, k_rot, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(v, v, 1e-6f);
    #undef TEST_INTERMEDIATE_ALLCLOSE
}

#endif