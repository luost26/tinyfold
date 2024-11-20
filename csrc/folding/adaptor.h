#ifndef FOLDING_ADAPTOR_H
#define FOLDING_ADAPTOR_H

#include <iostream>
#include <fstream>
#include <omp.h>
#include "../matrix.h"
#include "../linear.h"

struct AdaptorConfig {
    int c_s;
    int c_z;
    int sm_c_s;
    int sm_c_z;
    int esm_feats;
    int esm_attns;
    int esm_num_layers;
    int position_bins;
    int recycle_bins;

    AdaptorConfig(const std::string &cfg_path) {
        std::ifstream cfg_file(cfg_path);
        if (!cfg_file.is_open())
        {
            std::cerr << "Could not open config file" << std::endl;
            exit(1);
        }
        cfg_file >> c_s >> c_z >> sm_c_s >> sm_c_z >> esm_feats >> esm_attns >> esm_num_layers >> position_bins >> recycle_bins;
        cfg_file.close();
    }
};

struct AdaptorBuffer {
    int seqlen;
    AdaptorConfig cfg;

    matrix<float> s;
    matrix<float> z;

    matrix<float> sm_s;

    matrix<float> s_inplace_buffer;
    matrix<float> z_inplace_buffer;

    AdaptorBuffer(int seqlen, const AdaptorConfig &cfg):
        seqlen(seqlen),
        cfg(cfg),
        s(seqlen, cfg.c_s),
        z(seqlen * seqlen, cfg.c_z),
        sm_s(seqlen, cfg.sm_c_s),
        s_inplace_buffer(omp_get_max_threads(), cfg.c_s),
        z_inplace_buffer(omp_get_max_threads(), cfg.c_z)
    {}
};

struct Adaptor {
    AdaptorConfig cfg;

    matrix<float> esm_s_mlp_0_layernorm_weight;
    matrix<float> esm_s_mlp_0_layernorm_bias;
    matrix<float> esm_s_mlp_1_linear_weight;
    matrix<float> esm_s_mlp_1_linear_bias;
    // esm_s_mlp_2 is a ReLU layer with no weights
    matrix<float> esm_s_mlp_3_linear_weight;
    matrix<float> esm_s_mlp_3_linear_bias;

    matrix<float> esm_z_mlp_0_layernorm_weight;
    matrix<float> esm_z_mlp_0_layernorm_bias;
    matrix<float> esm_z_mlp_1_linear_weight;
    matrix<float> esm_z_mlp_1_linear_bias;
    // esm_z_mlp_2 is a ReLU layer with no weights
    matrix<float> esm_z_mlp_3_linear_weight;
    matrix<float> esm_z_mlp_3_linear_bias;

    matrix<float> embedding_weight;

    matrix<float> trunk_pairwise_positional_embedding;
    matrix<float> trunk_recycle_disto_weight;

    matrix<float> trunk_recycle_s_norm_weight;
    matrix<float> trunk_recycle_s_norm_bias;
    matrix<float> trunk_recycle_z_norm_weight;
    matrix<float> trunk_recycle_z_norm_bias;

    matrix<float> trunk_trunk2sm_s_weight;
    matrix<float> trunk_trunk2sm_s_bias;
    matrix<float> trunk_trunk2sm_z_weight;
    matrix<float> trunk_trunk2sm_z_bias;

    Adaptor(const AdaptorConfig &cfg, const std::string &dirpath):
        cfg(cfg),
        esm_s_mlp_0_layernorm_weight(cfg.esm_feats, 1),
        esm_s_mlp_0_layernorm_bias(cfg.esm_feats, 1),
        esm_s_mlp_1_linear_weight(cfg.c_s, cfg.esm_feats),
        esm_s_mlp_1_linear_bias(cfg.c_s, 1),
        esm_s_mlp_3_linear_weight(cfg.c_s, cfg.c_s),
        esm_s_mlp_3_linear_bias(cfg.c_s, 1),

        esm_z_mlp_0_layernorm_weight(cfg.esm_attns, 1),
        esm_z_mlp_0_layernorm_bias(cfg.esm_attns, 1),
        esm_z_mlp_1_linear_weight(cfg.c_z, cfg.esm_attns),
        esm_z_mlp_1_linear_bias(cfg.c_z, 1),
        esm_z_mlp_3_linear_weight(cfg.c_z, cfg.c_z),
        esm_z_mlp_3_linear_bias(cfg.c_z, 1),

        embedding_weight(20 + 3, cfg.c_s),
        trunk_pairwise_positional_embedding(cfg.position_bins * 2 + 2, cfg.c_z),
        trunk_recycle_disto_weight(cfg.recycle_bins, cfg.c_z),

        trunk_recycle_s_norm_weight(cfg.c_s, 1),
        trunk_recycle_s_norm_bias(cfg.c_s, 1),
        trunk_recycle_z_norm_weight(cfg.c_z, 1),
        trunk_recycle_z_norm_bias(cfg.c_z, 1),

        trunk_trunk2sm_s_weight(cfg.sm_c_s, cfg.c_s),
        trunk_trunk2sm_s_bias(cfg.sm_c_s, 1),
        trunk_trunk2sm_z_weight(cfg.sm_c_z, cfg.c_z),
        trunk_trunk2sm_z_bias(cfg.sm_c_z, 1)
    {
        std::cerr << "Loading weights for Adaptor from " << dirpath << std::endl;
        load_(esm_s_mlp_0_layernorm_weight, dirpath + "/esm_s_mlp.0.weight.bin");
        load_(esm_s_mlp_0_layernorm_bias, dirpath + "/esm_s_mlp.0.bias.bin");
        load_(esm_s_mlp_1_linear_weight, dirpath + "/esm_s_mlp.1.weight.bin");
        load_(esm_s_mlp_1_linear_bias, dirpath + "/esm_s_mlp.1.bias.bin");
        load_(esm_s_mlp_3_linear_weight, dirpath + "/esm_s_mlp.3.weight.bin");
        load_(esm_s_mlp_3_linear_bias, dirpath + "/esm_s_mlp.3.bias.bin");

        load_(esm_z_mlp_0_layernorm_weight, dirpath + "/esm_z_mlp.0.weight.bin");
        load_(esm_z_mlp_0_layernorm_bias, dirpath + "/esm_z_mlp.0.bias.bin");
        load_(esm_z_mlp_1_linear_weight, dirpath + "/esm_z_mlp.1.weight.bin");
        load_(esm_z_mlp_1_linear_bias, dirpath + "/esm_z_mlp.1.bias.bin");
        load_(esm_z_mlp_3_linear_weight, dirpath + "/esm_z_mlp.3.weight.bin");
        load_(esm_z_mlp_3_linear_bias, dirpath + "/esm_z_mlp.3.bias.bin");

        load_(embedding_weight, dirpath + "/embedding.weight.bin");
        load_(trunk_pairwise_positional_embedding, dirpath + "/trunk.pairwise_positional_embedding.embedding.weight.bin");
        load_(trunk_recycle_disto_weight, dirpath + "/trunk.recycle_disto.weight.bin");

        load_(trunk_recycle_s_norm_weight, dirpath + "/trunk.recycle_s_norm.weight.bin");
        load_(trunk_recycle_s_norm_bias, dirpath + "/trunk.recycle_s_norm.bias.bin");
        load_(trunk_recycle_z_norm_weight, dirpath + "/trunk.recycle_z_norm.weight.bin");
        load_(trunk_recycle_z_norm_bias, dirpath + "/trunk.recycle_z_norm.bias.bin");

        load_(trunk_trunk2sm_s_weight, dirpath + "/trunk.trunk2sm_s.weight.bin");
        load_(trunk_trunk2sm_s_bias, dirpath + "/trunk.trunk2sm_s.bias.bin");
        load_(trunk_trunk2sm_z_weight, dirpath + "/trunk.trunk2sm_z.weight.bin");
        load_(trunk_trunk2sm_z_bias, dirpath + "/trunk.trunk2sm_z.bias.bin");
    }

    AdaptorBuffer * create_buffer(int seqlen) const {
        return new AdaptorBuffer(seqlen, cfg);
    }

    void operator()(const matrix<float> &esm_s, const matrix<float> &esm_z, const matrix<int> &aatype, const matrix<int> &residx, AdaptorBuffer &buffer) {
        // esm_s_mlp
        fused_layer_norm_linear<ReLU>(esm_s, esm_s_mlp_0_layernorm_weight, esm_s_mlp_0_layernorm_bias, esm_s_mlp_1_linear_weight, esm_s_mlp_1_linear_bias, buffer.s);
        linear_(buffer.s, esm_s_mlp_3_linear_weight, esm_s_mlp_3_linear_bias, &buffer.s_inplace_buffer);
        #pragma omp parallel for
        for (int i = 0; i < buffer.s.n_rows; i ++) {
            for (int j = 0; j < buffer.s.n_cols; j ++) {
                *buffer.s(i, j) += *embedding_weight(*aatype(i, 0), j);
            }
        }

        // esm_z_mlp
        fused_layer_norm_linear<ReLU>(esm_z, esm_z_mlp_0_layernorm_weight, esm_z_mlp_0_layernorm_bias, esm_z_mlp_1_linear_weight, esm_z_mlp_1_linear_bias, buffer.z);
        linear_(buffer.z, esm_z_mlp_3_linear_weight, esm_z_mlp_3_linear_bias, &buffer.z_inplace_buffer);

        // No recycling
        for (int i = 0; i < buffer.s.n_rows; i ++) {
            for (int j = 0; j < buffer.s.n_cols; j ++) {
                *buffer.s(i, j) += *trunk_recycle_s_norm_bias(j, 0);
            }
        }
        linear(buffer.s, trunk_trunk2sm_s_weight, trunk_trunk2sm_s_bias, buffer.sm_s);

        #pragma omp parallel for
        for (int i = 0; i < buffer.z.n_rows; i ++) {
            int residx_i = *residx(i / buffer.s.n_rows, 0);
            int residx_j = *residx(i % buffer.s.n_rows, 0);
            int rpe_idx = residx_j - residx_i;
            if (rpe_idx < -cfg.position_bins) {
                rpe_idx = -cfg.position_bins;
            } else if (rpe_idx > cfg.position_bins) {
                rpe_idx = cfg.position_bins;
            }
            rpe_idx += cfg.position_bins + 1;

            for (int j = 0; j < buffer.z.n_cols; j ++) {
                *buffer.z(i, j) += *trunk_recycle_z_norm_bias(j, 0) + *trunk_recycle_disto_weight(0, j) + *trunk_pairwise_positional_embedding(rpe_idx, j);
            }
        }
        linear_(buffer.z, trunk_trunk2sm_z_weight, trunk_trunk2sm_z_bias, &buffer.z_inplace_buffer);
    }
};

Adaptor * load_adaptor(const std::string &dirpath) {
    AdaptorConfig cfg(dirpath + "/config.txt");
    return new Adaptor(cfg, dirpath);
}

#endif // FOLDING_ADAPTOR_H