#ifndef FOLDING_STRUCTURE_MODULE_H
#define FOLDING_STRUCTURE_MODULE_H

#include <iostream>
#include <fstream>
#include <memory>
#include "../matrix.h"
#include "linear.h"
#include "ipa.h"

constexpr float _QUAT_MULTIPLY[4][4][4] = {
    {{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, -1}},
    {{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, -1, 0}},
    {{0, 0, 1, 0}, {0, 0, 0, -1}, {1, 0, 0, 0}, {0, 1, 0, 0}},
    {{0, 0, 0, 1}, {0, 0, 1, 0}, {0, -1, 0, 0}, {1, 0, 0, 0}}
};

constexpr float _QUAT_MULTIPLY_BY_VEC[4][3][4] = {
    {{0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, -1}},
    {{1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, -1, 0}},
    {{0, 0, 0, -1}, {1, 0, 0, 0}, {0, 1, 0, 0}},
    {{0, 0, 1, 0}, {0, -1, 0, 0}, {1, 0, 0, 0}}
};


inline void affine_to_quat(float *r, float *quat) {
    float m00 = r[0], m01 = r[1], m02 = r[2];
    float m10 = r[4], m11 = r[5], m12 = r[6];
    float m20 = r[8], m21 = r[9], m22 = r[10];
    #define SQRT_RELU(x) (sqrtf(fmaxf(0.0f, x)))
    float q_abs[4] = {
        SQRT_RELU(1.0 + m00 + m11 + m22),
        SQRT_RELU(1.0 + m00 - m11 - m22),
        SQRT_RELU(1.0 - m00 + m11 - m22),
        SQRT_RELU(1.0 - m00 - m11 + m22)
    };

    int best_q_idx = 0;
    for (int i = 1; i < 4; i ++) {
        if (q_abs[i] > q_abs[best_q_idx]) {
            best_q_idx = i;
        }
    }

    float quat[4] = {0};
    #define UNPACK_QUAT(out,q1,q2,q3,q4) { out[0] = q1; out[1] = q2; out[2] = q3; out[3] = q4; }
    switch (best_q_idx) {
    case 0:
        UNPACK_QUAT(quat, q_abs[0] * q_abs[0], m21 - m12, m02 - m20, m10 - m01);
        break;
    case 1:
        UNPACK_QUAT(quat, m21 - m12, q_abs[1] * q_abs[1], m10 + m01, m02 + m20);
        break;
    case 2:
        UNPACK_QUAT(quat, m02 - m20, m10 + m01, q_abs[2] * q_abs[2], m12 + m21);
    default:
        UNPACK_QUAT(quat, m10 - m01, m20 + m02, m21 + m12, q_abs[3] * q_abs[3]);
        break;
    }
    #undef UNPACK_QUAT

    if (quat[0] < 0) {
        for (int i = 0; i < 4; i ++) {
            quat[i] = -quat[i];
        }
    }
}


inline void apply_q_update(float *r, const float *q_update) {
    float quat[4];
    affine_to_quat(r, quat);

    float new_quat[4] = {quat[0], quat[1], quat[2], quat[3]};
    for (int k = 0; k < 4; k ++) {
        for (int i = 0; i < 4; i ++) {
            for (int j = 0; j < 3; j ++) {
                new_quat[k] += _QUAT_MULTIPLY[i][j][k] * quat[i] * q_update[j];
            }
        }
    }
    
    float length = 0.0f;
    for (int i = 0; i < 4; i ++) {
        length += new_quat[i] * new_quat[i];
    }
    length = sqrtf(length);
    for (int i = 0; i < 4; i ++) {
        new_quat[i] /= length;
    }

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
        cfg_file >> c_s >> c_z >> c_ipa >> c_resnet >> no_blocks >> no_transition_layers >> no_resnet_blocks >> no_angles;
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

        bb_update_linear_weight(cfg.c_s, 6),
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

    inline void bb_update(StructureModuleBuffer &buffer) {
        for (int i = 0; i < buffer.seqlen; i ++) {
            float update_vec[6] = {
                *bb_update_linear_bias(0, 0),
                *bb_update_linear_bias(1, 0),
                *bb_update_linear_bias(2, 0),
                *bb_update_linear_bias(3, 0),
                *bb_update_linear_bias(4, 0),
                *bb_update_linear_bias(5, 0)
            };
            float *q_vec = update_vec, *k_vec = update_vec + 3;
            for (int j = 0; j < 6; j ++) {
                for (int k = 0; k < cfg.c_s; j ++) {
                    q_vec[j] += *buffer.s(i, k) * *bb_update_linear_weight(k, j);
                }
            }
        }
    }

    void operator()(const matrix<float> &s, const matrix<float> &z, StructureModuleBuffer &buffer) {
        layer_norm(z, layer_norm_z_weight, layer_norm_z_bias, buffer.z);        
        fused_layer_norm_linear(s, layer_norm_s_weight, layer_norm_s_bias, linear_in_weight, linear_in_bias, buffer.s);

        // for (int i = 0; i < cfg.no_blocks; i ++) {
        (*ipa)(buffer.s, buffer.z, buffer.r, buffer.s, *buffer.ipa_buffer, true);
        layer_norm(buffer.s, layer_norm_ipa_weight, layer_norm_ipa_bias, buffer.s);
        transition(buffer);

    }
};


StructureModule * load_structure_module(std::string dirpath) {
    auto ipa_dirpath = dirpath + "/ipa";
    auto cfg = StructureModuleConfig(dirpath + "/config.txt", {ipa_dirpath + "/config.txt"});
    return new StructureModule(cfg, dirpath, load_invariant_point_attention(ipa_dirpath));
}


#endif // FOLDING_STRUCTURE_MODULE_H