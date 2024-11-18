#ifndef PLM_TRANSFORMER_H
#define PLM_TRANSFORMER_H

#include <iostream>
#include <fstream>
#include "../matrix.h"
#include "../linear.h"


inline float rotary_angle(int s, int d, int dim) {
    const float inv_freq = 1.0f / powf(10000.0f, (float)((d * 2) % dim) / (float)dim);
    const float angle = (float)s * inv_freq;
    return angle;
}

inline void apply_rotary_embedding_(matrix<float> &x, int num_heads) {
    const int dim = x.n_cols;
    const int seqlen = x.n_rows / num_heads;
    const int half_dim = dim / 2;
    #pragma omp parallel for
    for (int i = 0; i < num_heads * seqlen; ++i) {
        const int head_idx = i / seqlen;
        const int seq_idx = i % seqlen;
        for (int j = 0; j < half_dim; j ++) {
            float angle_lo = rotary_angle(seq_idx, j, dim);
            float c_lo = cosf(angle_lo);
            float s_lo = sinf(angle_lo);
            float x_lo = *x(i, j) * c_lo - *x(i, j + half_dim) * s_lo;

            float angle_hi = rotary_angle(seq_idx, j + half_dim, dim);
            float c_hi = cosf(angle_hi);
            float s_hi = sinf(angle_hi);
            float x_hi = *x(i, j + half_dim) * c_hi + *x(i, j) * s_hi;

            *x(i, j) = x_lo;
            *x(i, j + half_dim) = x_hi;
        }
    }
}

struct TransformerConfig {
    int embed_dim;
    int num_heads;
    int kdim;
    int vdim;
    int ffn_embed_dim;

    int head_dim() const {
        return embed_dim / num_heads;
    }

    TransformerConfig(int embed_dim, int num_heads, int kdim, int vdim, int ffn_embed_dim):
        embed_dim(embed_dim),
        num_heads(num_heads),
        kdim(kdim),
        vdim(vdim),
        ffn_embed_dim(ffn_embed_dim)
    {}

    TransformerConfig(const std::string &cfg_path) {
        std::ifstream cfg_file(cfg_path);
        if (!cfg_file.is_open())
        {
            std::cerr << "Could not open config file" << std::endl;
            exit(1);
        }
        cfg_file >> embed_dim >> num_heads >> kdim >> vdim >> ffn_embed_dim;
        cfg_file.close();
    }
};

struct TransformerBuffer {
    int seqlen;
    TransformerConfig cfg;

    matrix<float> x;

    matrix<float> q;
    matrix<float> k;
    matrix<float> attn_weights;

    TransformerBuffer(int seqlen, const TransformerConfig &cfg):
        seqlen(seqlen),
        cfg(cfg),
        x(seqlen, cfg.embed_dim),
        q(cfg.num_heads * seqlen, cfg.head_dim()),
        k(cfg.num_heads * seqlen, cfg.head_dim()),
        attn_weights(cfg.num_heads * seqlen, seqlen)
    {}
};


struct TransformerLayer {
    TransformerConfig cfg;
    matrix<float> self_attn_layer_norm_weight;
    matrix<float> self_attn_layer_norm_bias;

    matrix<float> k_proj_weight;
    matrix<float> k_proj_bias;
    matrix<float> v_proj_weight;
    matrix<float> v_proj_bias;
    matrix<float> q_proj_weight;
    matrix<float> q_proj_bias;
    matrix<float> out_proj_weight;
    matrix<float> out_proj_bias;

    matrix<float> fc1_weight;
    matrix<float> fc1_bias;
    matrix<float> fc2_weight;
    matrix<float> fc2_bias;
    matrix<float> final_layer_norm_weight;
    matrix<float> final_layer_norm_bias;

    TransformerLayer(const TransformerConfig & cfg, const std::string &dirpath):
        cfg(cfg),
        self_attn_layer_norm_weight(cfg.embed_dim, 1),
        self_attn_layer_norm_bias(cfg.embed_dim, 1),
        k_proj_weight(cfg.embed_dim, cfg.kdim),
        k_proj_bias(cfg.embed_dim, 1),
        v_proj_weight(cfg.embed_dim, cfg.vdim),
        v_proj_bias(cfg.embed_dim, 1),
        q_proj_weight(cfg.embed_dim, cfg.embed_dim),
        q_proj_bias(cfg.embed_dim, 1),
        out_proj_weight(cfg.embed_dim, cfg.embed_dim),
        out_proj_bias(cfg.embed_dim, 1),
        fc1_weight(cfg.ffn_embed_dim, cfg.embed_dim),
        fc1_bias(cfg.ffn_embed_dim, 1),
        fc2_weight(cfg.embed_dim, cfg.ffn_embed_dim),
        fc2_bias(cfg.embed_dim, 1),
        final_layer_norm_weight(cfg.embed_dim, 1),
        final_layer_norm_bias(cfg.embed_dim, 1)
    {
        std::cerr << "Loading weights for TransformerLayer from " << dirpath << std::endl;
        load_(self_attn_layer_norm_weight, dirpath + "/self_attn_layer_norm.weight.bin");
        load_(self_attn_layer_norm_bias, dirpath + "/self_attn_layer_norm.bias.bin");
        load_(k_proj_weight, dirpath + "/self_attn.k_proj.weight.bin");
        load_(k_proj_bias, dirpath + "/self_attn.k_proj.bias.bin");
        load_(v_proj_weight, dirpath + "/self_attn.v_proj.weight.bin");
        load_(v_proj_bias, dirpath + "/self_attn.v_proj.bias.bin");
        load_(q_proj_weight, dirpath + "/self_attn.q_proj.weight.bin");
        load_(q_proj_bias, dirpath + "/self_attn.q_proj.bias.bin");
        mul_(q_proj_weight, 1.0f / sqrtf(cfg.head_dim()));
        mul_(q_proj_bias, 1.0f / sqrtf(cfg.head_dim()));

        load_(out_proj_weight, dirpath + "/self_attn.out_proj.weight.bin");
        load_(out_proj_bias, dirpath + "/self_attn.out_proj.bias.bin");
        load_(fc1_weight, dirpath + "/fc1.weight.bin");
        load_(fc1_bias, dirpath + "/fc1.bias.bin");
        load_(fc2_weight, dirpath + "/fc2.weight.bin");
        load_(fc2_bias, dirpath + "/fc2.bias.bin");
        load_(final_layer_norm_weight, dirpath + "/final_layer_norm.weight.bin");
        load_(final_layer_norm_bias, dirpath + "/final_layer_norm.bias.bin");
    }

    TransformerBuffer * create_buffer(int seqlen) const {
        return new TransformerBuffer(seqlen, cfg);
    }

    void apply_rotary_embedding() {

    }

    void operator() (const matrix<float> &x, TransformerBuffer &buffer) {
        layer_norm(x, self_attn_layer_norm_weight, self_attn_layer_norm_bias, buffer.x);
        attn_proj_linear(buffer.x, q_proj_weight, q_proj_bias, buffer.q);
        attn_proj_linear(buffer.x, k_proj_weight, k_proj_bias, buffer.k);
        apply_rotary_embedding_(buffer.q, cfg.num_heads);
        apply_rotary_embedding_(buffer.k, cfg.num_heads);
        bmm<true>(buffer.q, buffer.k, buffer.attn_weights);
        softmax_(buffer.attn_weights);

        attn_proj_linear(buffer.x, v_proj_weight, v_proj_bias, buffer.k);  // Use k buffer to save memory
        bmm(buffer.attn_weights, buffer.k, buffer.q);  // Use q buffer to save memory
        attn_out_linear(buffer.q, out_proj_weight, out_proj_bias, buffer.x);
    }
};


TransformerLayer * load_transformer_layer(const std::string &dirpath) {
    TransformerConfig cfg(dirpath + "/config.txt");
    return new TransformerLayer(cfg, dirpath);
}


#endif // PLM_TRANSFORMER_H