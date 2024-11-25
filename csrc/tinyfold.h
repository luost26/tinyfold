#ifndef TINYFOLD_H
#define TINYFOLD_H

#include <memory>
#include <string>
#include <cstdlib>
#include "plm/esm.h"
#include "folding/adaptor.h"
#include "folding/structure_module.h"
#include "folding/to_pdb.h"
#include "matrix.h"

struct TinyFoldConfig {
    bool memory_saving_mode = true;

    TinyFoldConfig() {
        if (std::getenv("NO_MEMORY_SAVING") != nullptr) {
            std::cout << "TinyFoldConfig: memory saving mode disabled" << std::endl;
            memory_saving_mode = false;
        }
    }
};

template <typename ESMTransformerWeightType>
struct TinyFold {
    std::unique_ptr<ESM<ESMTransformerWeightType>> esm;
    std::unique_ptr<matrix<float>> esm_s_combine_normalized;
    std::unique_ptr<Adaptor> adaptor;
    std::unique_ptr<StructureModule> structure_module;
    TinyFoldConfig config;
    TinyFold(ESM<ESMTransformerWeightType> *esm, matrix<float> *esm_s_combine_normalized, Adaptor *adaptor, StructureModule *structure_module, const TinyFoldConfig &config):
        esm(esm),
        esm_s_combine_normalized(esm_s_combine_normalized),
        adaptor(adaptor),
        structure_module(structure_module),
        config(config)
    {}

    std::string operator()(const std::string &seq) {
        matrix<int> esm_aatype = tokenize_esm_aatype(seq);
        if (esm_aatype.n_rows == 0) {
            std::cerr << "Empty/invalid sequence, ignoring" << std::endl;
            return "";
        }
        int seqlen = esm_aatype.n_rows;
        std::unique_ptr<ESMRepresentation> repr_out;
        if (config.memory_saving_mode) {
            repr_out.reset(new ESMRepresentation(
                seqlen,
                esm->cfg,
                *esm_s_combine_normalized,
                adaptor->esm_z_mlp_0_layernorm_weight,
                adaptor->esm_z_mlp_0_layernorm_bias,
                adaptor->esm_z_mlp_1_linear_weight,
                adaptor->esm_z_mlp_1_linear_bias));
        } else {
            repr_out.reset(new ESMRepresentation(seqlen, esm->cfg, *esm_s_combine_normalized));
        }

        {
            std::unique_ptr<ESMBuffer> esm_buffer(esm->create_buffer(esm_aatype));
            (*esm)(esm_aatype, *esm_buffer, repr_out.get());
        }

        matrix<int> folding_aatype = tokenize_folding_aatype(seq);
        matrix<int> residx = make_residx(seqlen);
        std::unique_ptr<AdaptorBuffer> adaptor_buffer(adaptor->create_buffer(seqlen));
        (*adaptor)(repr_out->s, repr_out->z, folding_aatype, residx, *adaptor_buffer);

        std::unique_ptr<StructureModuleBuffer> sm_buffer(structure_module->create_buffer(seqlen));
        (*structure_module)(adaptor_buffer->sm_s, adaptor_buffer->z, *sm_buffer);

        std::string pdb_out = to_pdb(sm_buffer->r, folding_aatype);
        return pdb_out;
    }
};

template <typename ESMTransformerWeightType>
TinyFold<ESMTransformerWeightType> * load_tinyfold(const std::string &dirpath, const TinyFoldConfig &config) {
    ESM<ESMTransformerWeightType> *esm = load_esm<ESMTransformerWeightType>(dirpath + "/esm");
    matrix<float> *esm_s_combine_normalized = new matrix<float>(dirpath + "/esm_s_combine_normalized.bin", esm->cfg.num_layers + 1, 1);
    Adaptor *adaptor = load_adaptor(dirpath + "/folding/adaptor");
    StructureModule *structure_module = load_structure_module(dirpath + "/folding/structure_module");
    return new TinyFold(esm, esm_s_combine_normalized, adaptor, structure_module, config);
}

template <typename ESMTransformerWeightType>
TinyFold<ESMTransformerWeightType> * load_tinyfold(const std::string &dirpath) {
    return load_tinyfold<ESMTransformerWeightType>(dirpath, TinyFoldConfig());
}


#endif  // TINYFOLD_H
