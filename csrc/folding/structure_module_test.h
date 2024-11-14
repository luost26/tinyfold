#ifndef FOLDING_STRUCTURE_MODULE_TEST_H
#define FOLDING_STRUCTURE_MODULE_TEST_H

#include <iostream>
#include <memory>
#include <random>
#include "structure_module.h"

void test_affine_quat_conversion() {
    std::cout << "====== Test Quaternion-Matrix Conversion ======" << std::endl;

    float quat[4];
    float r[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    // Random quaternion from 4d gaussian
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < 4; i++) {
        quat[i] = distribution(generator);
    }
    normalize_quat_(quat);
    standarize_quat_(quat);
    std::cout << quat[0] << quat[1] << quat[2] << quat[3] << std::endl;

    float quat_2[4];
    quat_to_affine(quat, r);
    affine_to_quat(r, quat_2);
    std::cout << quat_2[0] << quat_2[1] << quat_2[2] << quat_2[3] << std::endl;

    bool failed = false;
    for (int i = 0; i < 4; i++) {
        if (!(std::abs(quat[i] - quat_2[i]) < 1e-6)) {
            std::cout << "Test failed: quat[" << i << "] = " << quat[i] << " != quat_2[" << i << "] = " << quat_2[i] << std::endl;
            failed = true;
        }
    }
    if (!failed) {
        std::cout << "Test passed." << std::endl;
    }
}

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
    TEST_INTERMEDIATE_ALLCLOSE(s, s_ipa_7, 1e-5f);
    TEST_INTERMEDIATE_ALLCLOSE(r, affine_7, 2e-5f);
    #undef TEST_INTERMEDIATE_ALLCLOSE
}

#endif // FOLDING_STRUCTURE_MODULE_TEST_H
