#ifndef FOLDING_IPA_H
#define FOLDING_IPA_H

#include <iostream>
#include <fstream>
#include "../matrix.h"
#include "linear.h"

const float INF = 1e5f;
const float EPS = 1e-8f;


struct IPAConfig {
    int c_s;
    int c_z;
    int c_hidden;
    int no_heads;
    int no_qk_points;
    int no_v_points;

    IPAConfig(int c_s, int c_z, int c_hidden, int no_heads, int no_qk_points, int no_v_points):
        c_s(c_s),
        c_z(c_z),
        c_hidden(c_hidden),
        no_heads(no_heads),
        no_qk_points(no_qk_points),
        no_v_points(no_v_points)
    {}

    IPAConfig(const std::string &cfg_path)
    {
        std::ifstream cfg_file(cfg_path);
        if (!cfg_file.is_open())
        {
            std::cerr << "Could not open config file" << std::endl;
            exit(1);
        }
        cfg_file >> c_s >> c_z >> c_hidden >> no_heads >> no_qk_points >> no_v_points;
        cfg_file.close();
    }
};


struct IPAForwardBuffer {
    int seqlen;
    IPAConfig cfg;

    matrix<float> q;
    matrix<float> kv;

    matrix<float> q_pts;
    matrix<float> kv_pts;

    matrix<float> a;

    IPAForwardBuffer(int seqlen, const IPAConfig &cfg):
        seqlen(seqlen),
        cfg(cfg),
        q(seqlen, cfg.no_heads * cfg.c_hidden),
        kv(seqlen, cfg.no_heads * 2 * cfg.c_hidden),
        q_pts(seqlen, cfg.no_heads * cfg.no_qk_points * 3),
        kv_pts(seqlen, cfg.no_heads * (cfg.no_qk_points + cfg.no_v_points) * 3),
        a(cfg.no_heads * seqlen, seqlen)
    {}

    inline float * q_loc(int i, int h, int d) {
        return q.data + i * (cfg.no_heads * cfg.c_hidden) + h * cfg.c_hidden + d;
    }

    inline float * k_loc(int i, int h, int d) {
        return kv.data + i * (cfg.no_heads * 2 * cfg.c_hidden) + h * 2 * cfg.c_hidden + 0 * cfg.c_hidden + d;
    }

    inline float * v_loc(int i, int h, int d) {
        return kv.data + i * (cfg.no_heads * 2 * cfg.c_hidden) + h * 2 * cfg.c_hidden + 1 * cfg.c_hidden + d;
    }

};


struct InvariantPointAttention
{
    IPAConfig cfg;
    matrix<float> head_weights;

    matrix<float> linear_q_weight;
    matrix<float> linear_q_bias;
    matrix<float> linear_kv_weight;
    matrix<float> linear_kv_bias;

    matrix<float> linear_q_points_weight;
    matrix<float> linear_q_points_bias;
    matrix<float> linear_kv_points_weight;
    matrix<float> linear_kv_points_bias;

    matrix<float> linear_b_weight;
    matrix<float> linear_b_bias;

    matrix<float> linear_out_weight;
    matrix<float> linear_out_bias;

    InvariantPointAttention(const IPAConfig &cfg, const std::string &dirpath) :
        cfg(cfg),
        head_weights(cfg.no_heads, 1),
        linear_q_weight(cfg.c_hidden * cfg.no_heads, cfg.c_s),
        linear_q_bias(cfg.c_hidden * cfg.no_heads, 1),
        linear_kv_weight(2 * cfg.c_hidden * cfg.no_heads, cfg.c_s),
        linear_kv_bias(2 * cfg.c_hidden * cfg.no_heads, 1),
        linear_q_points_weight(cfg.no_heads * cfg.no_qk_points * 3, cfg.c_s),
        linear_q_points_bias(cfg.no_heads * cfg.no_qk_points * 3, 1),
        linear_kv_points_weight(cfg.no_heads * (cfg.no_qk_points + cfg.no_v_points) * 3, cfg.c_s),
        linear_kv_points_bias(cfg.no_heads * (cfg.no_qk_points + cfg.no_v_points) * 3, 1),
        linear_b_weight(cfg.no_heads, cfg.c_z),
        linear_b_bias(cfg.no_heads, 1),
        linear_out_weight(cfg.no_heads * (cfg.c_z + cfg.c_hidden + cfg.no_v_points) * 4, cfg.c_s),
        linear_out_bias(cfg.no_heads * (cfg.c_z + cfg.c_hidden + cfg.no_v_points) * 4, 1)
    {
        load_(head_weights, dirpath + "/head_weights.bin");
        load_(linear_q_weight, dirpath + "/linear_q.weight.bin");
        load_(linear_q_bias, dirpath + "/linear_q.bias.bin");
        load_(linear_kv_weight, dirpath + "/linear_kv.weight.bin");
        load_(linear_kv_bias, dirpath + "/linear_kv.bias.bin");
        load_(linear_q_points_weight, dirpath + "/linear_q_points.weight.bin");
        load_(linear_q_points_bias, dirpath + "/linear_q_points.bias.bin");
        load_(linear_kv_points_weight, dirpath + "/linear_kv_points.weight.bin");
        load_(linear_kv_points_bias, dirpath + "/linear_kv_points.bias.bin");
        load_(linear_b_weight, dirpath + "/linear_b.weight.bin");
        load_(linear_b_bias, dirpath + "/linear_b.bias.bin");
        load_(linear_out_weight, dirpath + "/linear_out.weight.bin");
        load_(linear_out_bias, dirpath + "/linear_out.bias.bin");
    }

    void operator()(const matrix<float> &s, const matrix<float> &z, const matrix<float> &r, matrix<float> &out, IPAForwardBuffer &buffer)
    {
        // s: (len, s_dim)
        // z: (len*len, z_dim)
        // r: (len, 4*4)
        linear(s, linear_q_weight, linear_q_bias, buffer.q);
        linear(s, linear_kv_weight, linear_kv_bias, buffer.kv);
    }
};

InvariantPointAttention * load_invariant_point_attention(std::string dirpath)
{
    IPAConfig cfg(dirpath + "/config.txt");
    return new InvariantPointAttention(cfg, dirpath);
}

std::ostream& operator<<(std::ostream &os, const InvariantPointAttention &A)
{
    #define SHOW_MATRIX_SIZE(name) {os << "  " << #name << " = (" << A.name.n_rows << ", " << A.name.n_cols << ")\n";};
    os << "InvariantPointAttention{" << std::endl;
    SHOW_MATRIX_SIZE(head_weights);
    SHOW_MATRIX_SIZE(linear_q_weight);
    SHOW_MATRIX_SIZE(linear_q_bias);
    SHOW_MATRIX_SIZE(linear_kv_weight);
    SHOW_MATRIX_SIZE(linear_kv_bias);
    SHOW_MATRIX_SIZE(linear_q_points_weight);
    SHOW_MATRIX_SIZE(linear_q_points_bias);
    SHOW_MATRIX_SIZE(linear_kv_points_weight);
    SHOW_MATRIX_SIZE(linear_kv_points_bias);
    SHOW_MATRIX_SIZE(linear_b_weight);
    SHOW_MATRIX_SIZE(linear_b_bias);
    SHOW_MATRIX_SIZE(linear_out_weight);
    SHOW_MATRIX_SIZE(linear_out_bias);
    os << "}" << std::endl;
    return os;
}

#endif // FOLDING_IPA_H
