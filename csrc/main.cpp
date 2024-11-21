#include <iostream>
#include "folding/ipa_test.h"
#include "folding/structure_module_test.h"
#include "folding/adaptor_test.h"
#include "folding/test.h"
#include "plm/transformer_test.h"
#include "plm/esm_test.h"
#include "tinyfold.h"
#include "matrix_q.h"
#include "utils/pseudo_quant.h"
#include "utils/pseudo_quant_test.h"

int main() {
    // test_ipa();
    // test_affine_quat_conversion();
    // test_structure_module();
    // test_adaptor();
    // test_folding();
    // test_transformer();
    // test_esm_small();
    // test_esm_full_3B();
    // benchmark_transformer();
    // test_pseudo_quantize();

    std::unique_ptr<TinyFold> tinyfold(load_tinyfold("../data/c_test/esmfold"));
    int block_size = 128;
    int num_bits = 4;
    #pragma omp parallel for
    for (int i = 0; i < tinyfold->esm->cfg.num_layers; i ++) {
        std::cerr << "Pseudo quantizing transformer layer #" << i + 1 << std::endl;
        pseudo_quantize_(tinyfold->esm->transformer_layers[i]->k_proj_weight, block_size, num_bits);
        pseudo_quantize_(tinyfold->esm->transformer_layers[i]->v_proj_weight, block_size, num_bits);
        pseudo_quantize_(tinyfold->esm->transformer_layers[i]->q_proj_weight, block_size, num_bits);
        pseudo_quantize_(tinyfold->esm->transformer_layers[i]->out_proj_weight, block_size, num_bits);
        pseudo_quantize_(tinyfold->esm->transformer_layers[i]->fc1_weight, block_size, num_bits);
        pseudo_quantize_(tinyfold->esm->transformer_layers[i]->fc2_weight, block_size, num_bits);
    }
    std::string seq("ASAWPEEKNYHQPAILNSSALRQIAEGTSISEMWQNDLQPLLIERYPGSPGSYAARQHIMQRIQRLQADWVLEIDTFLSQTPYGYRSFSNIISTLNPTAKRHLVLACHYDSKYFSHWNNRVFVGATDS");
    tinyfold->operator()(seq);
    return 0;
}
