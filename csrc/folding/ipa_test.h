#ifndef FOLDING_IPA_TEST_H
#define FOLDING_IPA_TEST_H

#include <iostream>
#include "ipa.h"
#include "../matrix.h"

#define TEST_ALLCLOSE(NAME,A,B,ATOL) { \
    float error; \
    if (!allclose(A, B, ATOL, &error)) { \
        std::cout << "Test " << NAME << ": FAIL, error = " << error << std::endl; \
        std::cout << "Expected:\n" << B << std::endl; \
        std::cout << "Got:\n" << A << std::endl; \
    } else { \
        std::cout << "Test " << NAME << ": PASS" << std::endl; \
    } \
}

void test_ipa() {
    std::cout << "====== Test IPA ======" << std::endl;
    int seqlen = 17;
    const std::string dirpath = "../data/c_test/ipa";
    std::cout << "seqlen = " << seqlen << std::endl;
    std::cout << "dirpath = " << dirpath << std::endl;

    InvariantPointAttention ipa = *load_invariant_point_attention(dirpath);
    matrix<float> s = matrix<float>(dirpath + "/input/s.bin", seqlen, ipa.cfg.c_s);
    matrix<float> z = matrix<float>(dirpath + "/input/z.bin", seqlen * seqlen, ipa.cfg.c_z);
    matrix<float> r = matrix<float>(dirpath + "/input/r.bin", seqlen, 4*4);
    IPAForwardBuffer buffer(seqlen, ipa.cfg);

    ipa(s, z, r, s, buffer);


    #define TEST_INTERMEDIATE_ALLCLOSE(NAME,ATOL) { \
        auto NAME##_ref = matrix<float>(dirpath + "/output/" #NAME ".bin", buffer.NAME.n_rows, buffer.NAME.n_cols); \
        TEST_ALLCLOSE(#NAME, buffer.NAME, NAME##_ref, ATOL); \
    }

    TEST_INTERMEDIATE_ALLCLOSE(q, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(k, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(v, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(q_pts, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(k_pts, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(v_pts, 1e-6f);
    TEST_INTERMEDIATE_ALLCLOSE(a, 1e-6f);
}

#endif // FOLDING_IPA_TEST_H