#ifndef FOLDING_IPA_H
#define FOLDING_IPA_H

#include "../matrix.h"
#include "linear.h"

struct InvariantPointAttention
{
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

    void operator()(matrix<float> &s, matrix<float> &z, matrix<float> &r, matrix<float> &out)
    {
        // s: (len, s_dim)
        // z: (len*len, z_dim)
        // r: (len, 3*3+3)
    }
};

#endif // FOLDING_IPA_H
