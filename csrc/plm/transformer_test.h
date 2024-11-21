#ifndef PLM_TRANSFORMER_TEST_H
#define PLM_TRANSFORMER_TEST_H

#include <iostream>
#include <memory>
#include <omp.h>
#include "transformer.h"

void test_transformer() {
    std::cout << "====== Test TransformerLayer ======" << std::endl;
    int seqlen = 17;
    const std::string dirpath = "../data/c_test/transformer";
    std::cout << "seqlen = " << seqlen << std::endl;
    std::cout << "dirpath = " << dirpath << std::endl;

    std::unique_ptr<TransformerLayer<matrix<float>>> transformer(load_transformer_layer(dirpath));
    matrix<float> x = matrix<float>(dirpath + "/input/x.bin", seqlen, transformer->cfg.embed_dim);
    std::unique_ptr<TransformerBuffer> buffer(transformer->create_buffer(seqlen));
    matrix<float> y(seqlen, transformer->cfg.embed_dim);

    (*transformer)(x, y, *buffer);

    #define TEST_INTERMEDIATE_ALLCLOSE(NAME,OUTNAME,ATOL) { \
        auto OUTNAME##_ref = matrix<float>(dirpath + "/output/" #OUTNAME ".bin", NAME.n_rows, NAME.n_cols); \
        TEST_ALLCLOSE(#NAME, NAME, OUTNAME##_ref, ATOL); \
    }
    // TEST_INTERMEDIATE_ALLCLOSE(q, q_rot, 1e-6f);
    // TEST_INTERMEDIATE_ALLCLOSE(k, k_rot, 1e-6f);
    // TEST_INTERMEDIATE_ALLCLOSE(v, v, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(buffer->attn_weights, attn_weights, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(buffer->x1, out_fc1, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(y, out, 1e-6f);
    #undef TEST_INTERMEDIATE_ALLCLOSE
}

void benchmark_transformer() {
    // Set OpenMP threads number to 4
    omp_set_num_threads(4);
    std::cout << "====== Benchmark TransformerLayer ======" << std::endl;
    int seqlen = 128;
    const std::string dirpath = "../data/c_test/esm_full_3B/transformer_0";
    std::cout << "seqlen = " << seqlen << std::endl;
    std::cout << "dirpath = " << dirpath << std::endl;

    std::unique_ptr<TransformerLayer<matrix<float>>> transformer(load_transformer_layer(dirpath));
    std::cout << "embed_dim: " << transformer->cfg.embed_dim << std::endl;
    std::cout << "num_heads: " << transformer->cfg.num_heads << std::endl;
    std::cout << "ffn_dim: " << transformer->cfg.ffn_embed_dim << std::endl;

    matrix<float> x = matrix<float>(seqlen, transformer->cfg.embed_dim);
    std::unique_ptr<TransformerBuffer> buffer(transformer->create_buffer(seqlen));
    matrix<float> y(seqlen, transformer->cfg.embed_dim);

    (*transformer)(x, y, *buffer);
}

#endif