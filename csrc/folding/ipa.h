#ifndef FOLDING_IPA_H
#define FOLDING_IPA_H

#include <iostream>
#include <fstream>
#include <cmath>
#include "../matrix.h"
#include "../linear.h"
#include "geometric.h"

const float INF = 1e5f;
const float EPS = 1e-8f;


inline void qk_inner_product_accumulate(const matrix<float> &query, const matrix<float> &key, float scale_factor, matrix<float> &out) {
    // q: (n_q, no_heads * c_hidden)
    // k: (n_k, no_heads * c_hidden)
    // out: (no_heads * n_q, n_k)
    int n_q = query.n_rows;
    int n_k = key.n_rows;
    int no_heads = out.n_rows / n_q;
    int c_hidden = query.n_cols / no_heads;
    for (int h = 0; h < no_heads; h++) {
        for (int i = 0; i < n_q; i++) {
            for (int j = 0; j < n_k; j++) {
                float sum = 0;
                for (int k = 0; k < c_hidden; k++) {
                    sum += *query(i, h * c_hidden + k) * *key(j, h * c_hidden + k);
                }
                *out(h * n_q + i, j) += sum * scale_factor;
            }
        }
    }
}


inline void pair_bias_linear_accumulate(const matrix<float> &z, const matrix<float> &weight, const matrix<float> &bias, float scale_factor, matrix<float> &out) {
    // z: (len*len, c_z)
    // weight: (no_heads, c_z)
    // bias: (no_heads, 1)
    // out: (no_heads * len, len)
    int len = out.n_cols;
    int no_heads = weight.n_rows;
    int c_z = weight.n_cols;
    for (int h = 0; h < no_heads; h ++) {
        for (int i = 0; i < len; i ++) {
            for (int j = 0; j < len; j ++) {
                float sum = 0;
                for (int k = 0; k < c_z; k ++) {
                    sum += *z(i * len + j, k) * *weight(h, k);
                }
                *out(h * len + i, j) += (sum + *bias(h, 0)) * scale_factor;
            }
        }
    }
}

inline float softplus(float x) {
    return log1p(exp(x));
}

inline void qk_pnts_accumulate(const matrix<float> &q_pts, const matrix<float> &k_pts, const matrix<float> &head_weights, matrix<float> &out) {
    // q_pts: (q_len, no_heads * no_qk_points * 3)
    // k_pts: (k_len, no_heads * no_qk_points * 3)
    // head_weights: (no_heads, 1)
    // out: (no_heads * q_len, k_len)
    int q_len = q_pts.n_rows;
    int k_len = k_pts.n_rows;
    int no_heads = head_weights.n_rows;
    int no_qk_points = q_pts.n_cols / (3 * no_heads);
    for (int h = 0; h < no_heads; h ++) {
        float hw = softplus(*head_weights(h, 0)) * sqrtf(1.0f / (3.0f * (no_qk_points * 9.0f / 2.0f)));
        for (int i = 0; i < q_len; i ++) {
            for (int j = 0; j < k_len; j ++) {
                float sum = 0.0f;
                for (int p = 0; p < no_qk_points; p ++) {
                    float d_x = *q_pts(i, h * no_qk_points * 3 + p * 3 + 0) - *k_pts(j, h * no_qk_points * 3 + p * 3 + 0);
                    float d_y = *q_pts(i, h * no_qk_points * 3 + p * 3 + 1) - *k_pts(j, h * no_qk_points * 3 + p * 3 + 1);
                    float d_z = *q_pts(i, h * no_qk_points * 3 + p * 3 + 2) - *k_pts(j, h * no_qk_points * 3 + p * 3 + 2);
                    sum += d_x * d_x + d_y * d_y + d_z * d_z;
                }
                sum *= hw * (-0.5f);
                *out(h * q_len + i, j) += sum;
            }
        }
    }
}

inline void compute_output(const matrix<float> &a, const matrix<float> &v, matrix<float> &out, int col_offset = 0) {
    // a: (no_heads * q_len, v_len)
    // v: (v_len, no_heads * c_hidden)
    // out: (q_len, [col_offset +] no_heads * c_hidden)
    int q_len = out.n_rows;
    int v_len = v.n_rows;
    int no_heads = a.n_rows / q_len;
    int c_hidden = v.n_cols / no_heads;
    for (int i = 0; i < q_len; i ++) {
        for (int h = 0; h < no_heads; h ++) {
            for (int k = 0; k < c_hidden; k ++) {
                float sum = 0.0f;
                for (int j = 0; j < v_len; j ++) {
                    sum += *a(h * q_len + i, j) * *v(j, h * c_hidden + k);
                }
                *out(i, col_offset + h * c_hidden + k) = sum;
            }
        }
    }
}

inline void compute_output_pts(const matrix<float> &a, const matrix<float> &v_pts, const matrix<float> &r, matrix<float> &out, int col_offset) {
    // a: (no_heads * q_len, v_len)
    // v_pts: (v_len, no_heads * no_v_points * 3)
    // r: (q_len, 4*4)
    // out: (q_len, [col_offset +] (3 * no_heads * no_v_points) + (no_heads * no_v_points))
    int q_len = out.n_rows;
    int v_len = v_pts.n_rows;
    int no_heads = a.n_rows / q_len;
    int no_v_points = v_pts.n_cols / (3 * no_heads);
    for (int i = 0; i < q_len; i ++) {
        for (int h = 0; h < no_heads; h ++) {
            for (int p = 0; p < no_v_points; p++) {
                float sum_crd[3] = {0.0f, 0.0f, 0.0f};
                for (int j = 0; j < v_len; j ++) {
                    float w = *a(h * q_len + i, j);
                    sum_crd[0] += w * *v_pts(j, h * no_v_points * 3 + p * 3 + 0);
                    sum_crd[1] += w * *v_pts(j, h * no_v_points * 3 + p * 3 + 1);
                    sum_crd[2] += w * *v_pts(j, h * no_v_points * 3 + p * 3 + 2);
                }
                invert_apply_affine(r(i, 0), sum_crd, sum_crd);
                float norm = sqrtf(sum_crd[0] * sum_crd[0] + sum_crd[1] * sum_crd[1] + sum_crd[2] * sum_crd[2] + EPS);
                *out(i, col_offset + 0 * no_heads * no_v_points + h * no_v_points + p) = sum_crd[0];
                *out(i, col_offset + 1 * no_heads * no_v_points + h * no_v_points + p) = sum_crd[1];
                *out(i, col_offset + 2 * no_heads * no_v_points + h * no_v_points + p) = sum_crd[2];
                *out(i, col_offset + 3 * no_heads * no_v_points + h * no_v_points + p) = norm;
            }
        }
    }
}

inline void compute_output_pair(const matrix<float> &a, const matrix<float> &z, matrix<float> &out, int col_offset) {
    // a: (no_heads * q_len, v_len)
    // z: (q_len*v_len, c_z)
    // out: (q_len, [col_offset +] no_heads * c_z)
    int q_len = out.n_rows;
    int v_len = a.n_cols;
    int no_heads = a.n_rows / q_len;
    int c_z = z.n_cols;
    for (int i = 0; i < q_len; i ++) {
        for (int h = 0; h < no_heads; h ++) {
            for (int k = 0; k < c_z; k ++) {
                float sum = 0.0f;
                for (int j = 0; j < v_len; j ++) {
                    float w = *a(h * q_len + i, j);
                    sum += w * *z(i * v_len + j, k);
                }
                *out(i, col_offset + h * c_z + k) = sum;
            }
        }
    }
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
    matrix<float> k;
    matrix<float> v;

    matrix<float> q_pts;
    matrix<float> k_pts;
    matrix<float> v_pts;

    matrix<float> a;
    matrix<float> o;

    IPAForwardBuffer(int seqlen, const IPAConfig &cfg):
        seqlen(seqlen),
        cfg(cfg),
        q(seqlen, cfg.no_heads * cfg.c_hidden),
        k(seqlen, cfg.no_heads * cfg.c_hidden),
        v(seqlen, cfg.no_heads * cfg.c_hidden),
        q_pts(seqlen, cfg.no_heads * cfg.no_qk_points * 3),
        k_pts(seqlen, cfg.no_heads * cfg.no_qk_points * 3),
        v_pts(seqlen, cfg.no_heads * cfg.no_v_points * 3),
        a(cfg.no_heads * seqlen, seqlen),
        o(seqlen, cfg.no_heads * (cfg.c_z + cfg.c_hidden + cfg.no_v_points * 4))
    {}

    inline float * q_loc(int i, int h, int d) {
        return q.data + i * (cfg.no_heads * cfg.c_hidden) + h * cfg.c_hidden + d;
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
    matrix<float> linear_k_points_weight;
    matrix<float> linear_k_points_bias;
    matrix<float> linear_v_points_weight;
    matrix<float> linear_v_points_bias;

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
        linear_k_points_weight(cfg.no_heads * cfg.no_qk_points * 3, cfg.c_s),
        linear_k_points_bias(cfg.no_heads * cfg.no_qk_points * 3, 1),
        linear_v_points_weight(cfg.no_heads * cfg.no_v_points * 3, cfg.c_s),
        linear_v_points_bias(cfg.no_heads * cfg.no_v_points * 3, 1),

        linear_b_weight(cfg.no_heads, cfg.c_z),
        linear_b_bias(cfg.no_heads, 1),
        linear_out_weight(cfg.c_s, cfg.no_heads * (cfg.c_z + cfg.c_hidden + cfg.no_v_points * 4)),
        linear_out_bias(cfg.c_s, 1)
    {
        std::cerr << "Loading weights for InvariantPointAttention from " << dirpath << std::endl;
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
                    } 
                    *linear_k_bias(i * cfg.c_hidden + j, 0) = *kv_b_raw(i * 2 * cfg.c_hidden + 0 * cfg.c_hidden + j, 0);
                    *linear_v_bias(i * cfg.c_hidden + j, 0) = *kv_b_raw(i * 2 * cfg.c_hidden + 1 * cfg.c_hidden + j, 0);
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
                }
                *linear_q_points_bias(i * 3 + 0, 0) = *q_pts_b_raw(grpsize * 0 + i, 0);
                *linear_q_points_bias(i * 3 + 1, 0) = *q_pts_b_raw(grpsize * 1 + i, 0);
                *linear_q_points_bias(i * 3 + 2, 0) = *q_pts_b_raw(grpsize * 2 + i, 0);
            }
        }

        /*
           Original output channels layout: (3, no_heads, no_qk_points + no_v_points)
           We rearrange it to (no_heads, no_qk_points, 3) and (no_heads, no_v_points, 3)
         */
        {
            matrix<float> kv_pts_w(dirpath + "/linear_kv_points.weight.bin", cfg.no_heads * (cfg.no_qk_points + cfg.no_v_points) * 3, cfg.c_s);
            matrix<float> kv_pts_b(dirpath + "/linear_kv_points.bias.bin", cfg.no_heads * (cfg.no_qk_points + cfg.no_v_points) * 3, 1);
            int sum_kv_pts = cfg.no_qk_points + cfg.no_v_points;
            for (int i = 0; i < cfg.no_heads; i ++) {
                for (int j = 0; j < cfg.no_qk_points; j ++) {
                    for (int k = 0; k < cfg.c_s; k ++) {
                        *linear_k_points_weight(i * cfg.no_qk_points * 3 + j * 3 + 0, k) = *kv_pts_w(0 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + j, k);
                        *linear_k_points_weight(i * cfg.no_qk_points * 3 + j * 3 + 1, k) = *kv_pts_w(1 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + j, k);
                        *linear_k_points_weight(i * cfg.no_qk_points * 3 + j * 3 + 2, k) = *kv_pts_w(2 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + j, k);
                    }
                    *linear_k_points_bias(i * cfg.no_qk_points * 3 + j * 3 + 0, 0) = *kv_pts_b(0 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + j, 0);
                    *linear_k_points_bias(i * cfg.no_qk_points * 3 + j * 3 + 1, 0) = *kv_pts_b(1 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + j, 0);
                    *linear_k_points_bias(i * cfg.no_qk_points * 3 + j * 3 + 2, 0) = *kv_pts_b(2 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + j, 0);
                }
                for (int j = 0; j < cfg.no_v_points; j ++) {
                    for (int k = 0; k < cfg.c_s; k ++) {
                        *linear_v_points_weight(i * cfg.no_v_points * 3 + j * 3 + 0, k) = *kv_pts_w(0 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + cfg.no_qk_points + j, k);
                        *linear_v_points_weight(i * cfg.no_v_points * 3 + j * 3 + 1, k) = *kv_pts_w(1 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + cfg.no_qk_points + j, k);
                        *linear_v_points_weight(i * cfg.no_v_points * 3 + j * 3 + 2, k) = *kv_pts_w(2 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + cfg.no_qk_points + j, k);
                    }
                    *linear_v_points_bias(i * cfg.no_v_points * 3 + j * 3 + 0, 0) = *kv_pts_b(0 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + cfg.no_qk_points + j, 0);
                    *linear_v_points_bias(i * cfg.no_v_points * 3 + j * 3 + 1, 0) = *kv_pts_b(1 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + cfg.no_qk_points + j, 0);
                    *linear_v_points_bias(i * cfg.no_v_points * 3 + j * 3 + 2, 0) = *kv_pts_b(2 * cfg.no_heads * sum_kv_pts + i * sum_kv_pts + cfg.no_qk_points + j, 0);
                }
            }
        }

        load_(linear_b_weight, dirpath + "/linear_b.weight.bin");
        load_(linear_b_bias, dirpath + "/linear_b.bias.bin");
        load_(linear_out_weight, dirpath + "/linear_out.weight.bin");
        load_(linear_out_bias, dirpath + "/linear_out.bias.bin");

        std::cerr << "InvariantPointAttention weights loaded." << std::endl;
    }

    IPAForwardBuffer * create_buffer(int seqlen) const {
        return new IPAForwardBuffer(seqlen, cfg);
    }

    void operator()(const matrix<float> &s, const matrix<float> &z, const matrix<float> &r, matrix<float> &out, IPAForwardBuffer &buffer, bool residual)
    {
        // s: (len, s_dim)
        // z: (len*len, z_dim)
        // r: (len, 4*4)
        linear(s, linear_q_weight, linear_q_bias, buffer.q);
        linear(s, linear_k_weight, linear_k_bias, buffer.k);
        linear(s, linear_v_weight, linear_v_bias, buffer.v);

        // q_pts: (len, no_heads * no_qk_points * 3)
        // k_pts: (len, no_heads * no_qk_points * 3)
        linear(s, linear_q_points_weight, linear_q_points_bias, buffer.q_pts);
        linear(s, linear_k_points_weight, linear_k_points_bias, buffer.k_pts);
        for (int i = 0; i < buffer.seqlen; i++) {
            for (int j = 0; j < cfg.no_heads * cfg.no_qk_points; j++) {
                apply_affine(r(i, 0), buffer.q_pts(i, j * 3), buffer.q_pts(i, j * 3));
                apply_affine(r(i, 0), buffer.k_pts(i, j * 3), buffer.k_pts(i, j * 3));
            }
        }
        linear(s, linear_v_points_weight, linear_v_points_bias, buffer.v_pts);
        for (int i = 0; i < buffer.seqlen; i++) {
            for (int j = 0; j < cfg.no_heads * cfg.no_v_points; j++) {
                apply_affine(r(i, 0), buffer.v_pts(i, j * 3), buffer.v_pts(i, j * 3));
            }
        }

        zero_(buffer.a);
        qk_inner_product_accumulate(buffer.q, buffer.k, 1.0f / sqrtf(3.0f * cfg.c_hidden), buffer.a);
        pair_bias_linear_accumulate(z, linear_b_weight, linear_b_bias, sqrtf(1.0f / 3.0f), buffer.a);
        qk_pnts_accumulate(buffer.q_pts, buffer.k_pts, head_weights, buffer.a);
        softmax_(buffer.a);

        compute_output(buffer.a, buffer.v, buffer.o, 0);
        compute_output_pts(buffer.a, buffer.v_pts, r, buffer.o, cfg.no_heads * cfg.c_hidden);
        compute_output_pair(buffer.a, z, buffer.o, cfg.no_heads * cfg.c_hidden + cfg.no_heads * cfg.no_v_points * 4);

        if (residual) {
            linear_residual(buffer.o, linear_out_weight, linear_out_bias, out);
        } else {
            linear(buffer.o, linear_out_weight, linear_out_bias, out);
        }
    }
};

InvariantPointAttention * load_invariant_point_attention(const std::string &dirpath)
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
    SHOW_MATRIX_SIZE(linear_b_weight);
    SHOW_MATRIX_SIZE(linear_b_bias);
    SHOW_MATRIX_SIZE(linear_out_weight);
    SHOW_MATRIX_SIZE(linear_out_bias);
    os << "}" << std::endl;
    return os;
}

#endif // FOLDING_IPA_H
