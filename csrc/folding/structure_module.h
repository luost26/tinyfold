#ifndef FOLDING_STRUCTURE_MODULE_H
#define FOLDING_STRUCTURE_MODULE_H

#include <iostream>
#include <fstream>
#include <memory>
#include "../matrix.h"
#include "../kernels.h"
#include "geometric.h"
#include "ipa.h"


inline void apply_qt_update(const float *r, const float *qt_update, float *r_new) {
    float trans_update[3] = {qt_update[3], qt_update[4], qt_update[5]};
    apply_affine_rotation_only(r, trans_update, trans_update);
    r_new[3] = r[3] + trans_update[0];
    r_new[7] = r[7] + trans_update[1];
    r_new[11] = r[11] + trans_update[2];

    float quat_update[4] = {1.0, qt_update[0], qt_update[1], qt_update[2]};
    normalize_quat_(quat_update);
    standarize_quat_(quat_update);
    float affine_update[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 0,
    };
    quat_to_affine(quat_update, affine_update);
    compose_rotation(r, affine_update, r_new);
}

struct StructureModuleConfig {
    int c_s;
    int c_z;
    int c_ipa;
    int c_resnet;
    int no_blocks;
    int no_transition_layers;
    int no_resnet_blocks;
    int no_angles;
    int trans_scale_factor;
    IPAConfig ipa_cfg;

    StructureModuleConfig(const std::string &path, const IPAConfig &ipa_cfg):
        ipa_cfg(ipa_cfg)
    {
        std::ifstream cfg_file(path);
        if (!cfg_file.is_open())
        {
            std::cerr << "Could not open config file" << std::endl;
            exit(1);
        }
        cfg_file >> c_s >> c_z >> c_ipa >> c_resnet >> no_blocks >> no_transition_layers >> no_resnet_blocks >> no_angles >> trans_scale_factor;
    }
};

struct StructureModuleBuffer {
    int seqlen;
    StructureModuleConfig cfg;
    std::unique_ptr<IPAForwardBuffer> ipa_buffer;

    matrix<float> s;
    matrix<float> s_1;
    matrix<float> s_2;
    matrix<float> z;
    matrix<float> r;

    StructureModuleBuffer(int seqlen, const StructureModuleConfig & cfg, IPAForwardBuffer * ipa_buffer):
        seqlen(seqlen),
        cfg(cfg),
        ipa_buffer(ipa_buffer),
        s(seqlen, cfg.c_s),
        s_1(seqlen, cfg.c_s),
        s_2(seqlen, cfg.c_s),
        z(seqlen * seqlen, cfg.c_z),
        r(seqlen, 4 * 4)
    {
        // identity affine matrix
        for (int i = 0; i < seqlen; i++) {
            *r(i, 0) = 1;
            *r(i, 5) = 1;
            *r(i, 10) = 1;
            *r(i, 15) = 1;
        }
    }
};

struct StructureModule {
    StructureModuleConfig cfg;
    std::unique_ptr<InvariantPointAttention> ipa;

    matrix<float> layer_norm_s_weight;
    matrix<float> layer_norm_s_bias;
    matrix<float> layer_norm_z_weight;
    matrix<float> layer_norm_z_bias;
    matrix<float> linear_in_weight;
    matrix<float> linear_in_bias;

    matrix<float> layer_norm_ipa_weight;
    matrix<float> layer_norm_ipa_bias;

    matrix<float> transition_layers_0_linear_1_weight;
    matrix<float> transition_layers_0_linear_1_bias;
    matrix<float> transition_layers_0_linear_2_weight;
    matrix<float> transition_layers_0_linear_2_bias;
    matrix<float> transition_layers_0_linear_3_weight;
    matrix<float> transition_layers_0_linear_3_bias;
    matrix<float> transition_layer_norm_weight;
    matrix<float> transition_layer_norm_bias;

    matrix<float> bb_update_linear_weight;
    matrix<float> bb_update_linear_bias;

    StructureModule(const StructureModuleConfig &cfg, const std::string &dirpath, InvariantPointAttention *ipa):
        cfg(cfg),
        ipa(ipa),
        layer_norm_s_weight(cfg.c_s, 1),
        layer_norm_s_bias(cfg.c_s, 1),
        layer_norm_z_weight(cfg.c_z, 1),
        layer_norm_z_bias(cfg.c_z, 1),
        linear_in_weight(cfg.c_s, cfg.c_s),
        linear_in_bias(cfg.c_s, 1),
        layer_norm_ipa_weight(cfg.c_s, 1),
        layer_norm_ipa_bias(cfg.c_s, 1),

        transition_layers_0_linear_1_weight(cfg.c_s, cfg.c_s),
        transition_layers_0_linear_1_bias(cfg.c_s, 1),
        transition_layers_0_linear_2_weight(cfg.c_s, cfg.c_s),
        transition_layers_0_linear_2_bias(cfg.c_s, 1),
        transition_layers_0_linear_3_weight(cfg.c_s, cfg.c_s),
        transition_layers_0_linear_3_bias(cfg.c_s, 1),
        transition_layer_norm_weight(cfg.c_s, 1),
        transition_layer_norm_bias(cfg.c_s, 1),

        bb_update_linear_weight(6, cfg.c_s),
        bb_update_linear_bias(6, 1)
    {
        std::cerr << "Loading weights for StructureModule from " << dirpath << std::endl;
        load_(layer_norm_s_weight, dirpath + "/layer_norm_s.weight.bin");
        load_(layer_norm_s_bias, dirpath + "/layer_norm_s.bias.bin");
        load_(layer_norm_z_weight, dirpath + "/layer_norm_z.weight.bin");
        load_(layer_norm_z_bias, dirpath + "/layer_norm_z.bias.bin");
        load_(linear_in_weight, dirpath + "/linear_in.weight.bin");
        load_(linear_in_bias, dirpath + "/linear_in.bias.bin");
        load_(layer_norm_ipa_weight, dirpath + "/layer_norm_ipa.weight.bin");
        load_(layer_norm_ipa_bias, dirpath + "/layer_norm_ipa.bias.bin");

        load_(transition_layers_0_linear_1_weight, dirpath + "/transition.layers.0.linear_1.weight.bin");
        load_(transition_layers_0_linear_1_bias, dirpath + "/transition.layers.0.linear_1.bias.bin");
        load_(transition_layers_0_linear_2_weight, dirpath + "/transition.layers.0.linear_2.weight.bin");
        load_(transition_layers_0_linear_2_bias, dirpath + "/transition.layers.0.linear_2.bias.bin");
        load_(transition_layers_0_linear_3_weight, dirpath + "/transition.layers.0.linear_3.weight.bin");
        load_(transition_layers_0_linear_3_bias, dirpath + "/transition.layers.0.linear_3.bias.bin");
        load_(transition_layer_norm_weight, dirpath + "/transition.layer_norm.weight.bin");
        load_(transition_layer_norm_bias, dirpath + "/transition.layer_norm.bias.bin");

        load_(bb_update_linear_weight, dirpath + "/bb_update.linear.weight.bin");
        load_(bb_update_linear_bias, dirpath + "/bb_update.linear.bias.bin");
    }

    StructureModuleBuffer * create_buffer(int seqlen) const {
        return new StructureModuleBuffer(seqlen, cfg, ipa->create_buffer(seqlen));
    }

    inline void transition(StructureModuleBuffer &buffer) {
        // Fused transition layer
        linear<ReLU>(buffer.s, transition_layers_0_linear_1_weight, transition_layers_0_linear_1_bias, buffer.s_1);
        linear<ReLU>(buffer.s_1, transition_layers_0_linear_2_weight, transition_layers_0_linear_2_bias, buffer.s_2);
        linear(buffer.s_2, transition_layers_0_linear_3_weight, transition_layers_0_linear_3_bias, buffer.s_1);
        add_(buffer.s, buffer.s_1);
        layer_norm(buffer.s, transition_layer_norm_weight, transition_layer_norm_bias, buffer.s);
    }

    inline void update_rigids(StructureModuleBuffer &buffer) {
        for (int i = 0; i < buffer.seqlen; i ++) {
            float update_vec[6] = {
                *bb_update_linear_bias(0, 0),
                *bb_update_linear_bias(1, 0),
                *bb_update_linear_bias(2, 0),
                *bb_update_linear_bias(3, 0),
                *bb_update_linear_bias(4, 0),
                *bb_update_linear_bias(5, 0)
            };
            for (int j = 0; j < 6; j ++) {
                for (int k = 0; k < cfg.c_s; k ++) {
                    update_vec[j] += *buffer.s(i, k) * *bb_update_linear_weight(j, k);
                }
            }
            apply_qt_update(buffer.r(i, 0), update_vec, buffer.r(i, 0));
        }
    }

    void operator()(const matrix<float> &s, const matrix<float> &z, StructureModuleBuffer &buffer) {
        layer_norm(z, layer_norm_z_weight, layer_norm_z_bias, buffer.z);        
        fused_layer_norm_linear(s, layer_norm_s_weight, layer_norm_s_bias, linear_in_weight, linear_in_bias, buffer.s);

        for (int i = 0; i < cfg.no_blocks; i ++) {
            (*ipa)(buffer.s, buffer.z, buffer.r, buffer.s, *buffer.ipa_buffer, true);
            layer_norm(buffer.s, layer_norm_ipa_weight, layer_norm_ipa_bias, buffer.s);
            transition(buffer);
            update_rigids(buffer);
        }

        for (int i = 0; i < buffer.seqlen; i ++) {
            *buffer.r(i, 3) *= cfg.trans_scale_factor;
            *buffer.r(i, 7) *= cfg.trans_scale_factor;
            *buffer.r(i, 11) *= cfg.trans_scale_factor;
        }
    }
};


StructureModule * load_structure_module(std::string dirpath) {
    auto ipa_dirpath = dirpath + "/ipa";
    auto cfg = StructureModuleConfig(dirpath + "/config.txt", {ipa_dirpath + "/config.txt"});
    return new StructureModule(cfg, dirpath, load_invariant_point_attention(ipa_dirpath));
}


#endif // FOLDING_STRUCTURE_MODULE_H