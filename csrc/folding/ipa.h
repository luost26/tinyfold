#ifndef FOLDING_IPA_H
#define FOLDING_IPA_H

#include <iostream>
#include <fstream>
#include "../matrix.h"
#include "linear.h"

const float INF = 1e5f;
const float EPS = 1e-8f;


inline void apply_affine(const float * affine_4x4, const float * vec3, float * out3) {
    float x = affine_4x4[0] * vec3[0] + affine_4x4[1] * vec3[1] + affine_4x4[2] * vec3[2] + affine_4x4[3];
    float y = affine_4x4[4] * vec3[0] + affine_4x4[5] * vec3[1] + affine_4x4[6] * vec3[2] + affine_4x4[7];
    float z = affine_4x4[8] * vec3[0] + affine_4x4[9] * vec3[1] + affine_4x4[10] * vec3[2] + affine_4x4[11];
    out3[0] = x;
    out3[1] = y;
    out3[2] = z;
}


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
    matrix<float> k;
    matrix<float> v;

    matrix<float> q_pts;
    matrix<float> kv_pts;

    matrix<float> a;

    IPAForwardBuffer(int seqlen, const IPAConfig &cfg):
        seqlen(seqlen),
        cfg(cfg),
        q(seqlen, cfg.no_heads * cfg.c_hidden),
        kv(seqlen, cfg.no_heads * 2 * cfg.c_hidden),
        k(seqlen, cfg.no_heads * cfg.c_hidden),
        v(seqlen, cfg.no_heads * cfg.c_hidden),
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
    matrix<float> linear_k_weight;
    matrix<float> linear_k_bias;
    matrix<float> linear_v_weight;
    matrix<float> linear_v_bias;

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
        linear_k_weight(cfg.c_hidden * cfg.no_heads, cfg.c_s),
        linear_k_bias(cfg.c_hidden * cfg.no_heads, 1),
        linear_v_weight(cfg.c_hidden * cfg.no_heads, cfg.c_s),
        linear_v_bias(cfg.c_hidden * cfg.no_heads, 1),

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
        {
            matrix<float> kv_w_raw(dirpath + "/linear_kv.weight.bin", 2 * cfg.c_hidden * cfg.no_heads, cfg.c_s);
            matrix<float> kv_b_raw(dirpath + "/linear_kv.bias.bin", 2 * cfg.c_hidden * cfg.no_heads, 1);
            for (int i = 0; i < cfg.no_heads; i ++) {
                for (int j = 0; j < cfg.c_hidden; j ++) {
                    for (int k = 0; k < cfg.c_s; k ++) {
                        *linear_k_weight(i * cfg.c_hidden + j, k) = *kv_w_raw(i * 2 * cfg.c_hidden + 0 * cfg.c_hidden + j, k);
                        *linear_v_weight(i * cfg.c_hidden + j, k) = *kv_w_raw(i * 2 * cfg.c_hidden + 1 * cfg.c_hidden + j, k);
                        *linear_k_bias(i * cfg.c_hidden + j, 0) = *kv_b_raw(i * 2 * cfg.c_hidden + 0 * cfg.c_hidden + j, 0);
                        *linear_v_bias(i * cfg.c_hidden + j, 0) = *kv_b_raw(i * 2 * cfg.c_hidden + 1 * cfg.c_hidden + j, 0);
                    } 
                }
            }
        }

        /* 
          The output channel layout of the original weight and bias is (3, no_heads, no_qk_points).
          We rearrange it to (no_heads, no_qk_points, 3).
         */
        {
            matrix<float> q_pts_w_raw(dirpath + "/linear_q_points.weight.bin", linear_q_points_weight.n_rows, linear_q_points_weight.n_cols);
            matrix<float> q_pts_b_raw(dirpath + "/linear_q_points.bias.bin", linear_q_points_bias.n_rows, linear_q_points_bias.n_cols);
            int grpsize = linear_q_points_weight.n_rows / 3;
            for (int i = 0; i < grpsize; i ++) {
                for (int j = 0; j < linear_q_points_weight.n_cols; j ++) {
                    *linear_q_points_weight(i * 3 + 0, j) = *q_pts_w_raw(grpsize * 0 + i, j);
                    *linear_q_points_weight(i * 3 + 1, j) = *q_pts_w_raw(grpsize * 1 + i, j);
                    *linear_q_points_weight(i * 3 + 2, j) = *q_pts_w_raw(grpsize * 2 + i, j);
                    *linear_q_points_bias(i * 3 + 0, 0) = *q_pts_b_raw(grpsize * 0 + i, 0);
                    *linear_q_points_bias(i * 3 + 1, 0) = *q_pts_b_raw(grpsize * 1 + i, 0);
                    *linear_q_points_bias(i * 3 + 2, 0) = *q_pts_b_raw(grpsize * 2 + i, 0);
                }
            }
        }
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
        linear(s, linear_k_weight, linear_k_bias, buffer.k);
        linear(s, linear_v_weight, linear_v_bias, buffer.v);

        // q_pts: (len, no_heads * no_qk_points * 3)
        linear(s, linear_q_points_weight, linear_q_points_bias, buffer.q_pts);
        for (int i = 0; i < buffer.seqlen; i++) {
            for (int j = 0; j < cfg.no_heads * cfg.no_qk_points; j++) {
                apply_affine(r(i, 0), buffer.q_pts(i, j * 3), buffer.q_pts(i, j * 3));
            }
        }
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
    SHOW_MATRIX_SIZE(linear_k_weight);
    SHOW_MATRIX_SIZE(linear_k_bias);
    SHOW_MATRIX_SIZE(linear_v_weight);
    SHOW_MATRIX_SIZE(linear_v_bias);
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
