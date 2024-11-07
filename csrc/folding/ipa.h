#ifndef FOLDING_IPA_H
#define FOLDING_IPA_H

#include <iostream>
#include <fstream>
#include "../matrix.h"
#include "linear.h"

const float INF = 1e5f;
const float EPS = 1e-8f;

struct InvariantPointAttention
{
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

    InvariantPointAttention(std::string dirpath, int c_s, int c_z, int c_hidden, int no_heads, int no_qk_points, int no_v_points) :
        head_weights(no_heads, 1),
        linear_q_weight(c_hidden * no_heads, c_s),
        linear_q_bias(c_hidden * no_heads, 1),
        linear_kv_weight(2 * c_hidden * no_heads, c_s),
        linear_kv_bias(2 * c_hidden * no_heads, 1),
        linear_q_points_weight(no_heads * no_qk_points * 3, c_s),
        linear_q_points_bias(no_heads * no_qk_points * 3, 1),
        linear_kv_points_weight(no_heads * (no_qk_points + no_v_points) * 3, c_s),
        linear_kv_points_bias(no_heads * (no_qk_points + no_v_points) * 3, 1),
        linear_b_weight(no_heads, c_z),
        linear_b_bias(no_heads, 1),
        linear_out_weight(no_heads * (c_z + c_hidden + no_v_points) * 4, c_s),
        linear_out_bias(no_heads * (c_z + c_hidden + no_v_points) * 4, 1)
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

    void operator()(matrix<float> &s, matrix<float> &z, matrix<float> &r, matrix<float> &out)
    {
        // s: (len, s_dim)
        // z: (len*len, z_dim)
        // r: (len, 3*3+3)
    }
};

InvariantPointAttention * load_invariant_point_attention(std::string dirpath)
{
    int c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points;

    std::ifstream cfg_file(dirpath + "/config.txt");
    if (!cfg_file.is_open())
    {
        std::cerr << "Could not open config file" << std::endl;
        exit(1);
    }
    cfg_file >> c_s >> c_z >> c_hidden >> no_heads >> no_qk_points >> no_v_points;
    cfg_file.close();

    return new InvariantPointAttention(dirpath, c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points);
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
