#ifndef PLM_TRANSFORMER_H
#define PLM_TRANSFORMER_H

#include <iostream>
#include <fstream>
#include <chrono>
#include "../matrix.h"
#include "../kernels.h"
#include "../utils/timer.h"
#include "transformer_kernels.h"

struct TransformerConfig {
    int embed_dim;  // 2560
    int num_heads;  // 40
    int kdim;  // 2560
    int vdim;  // 2560
    int ffn_embed_dim;  // 10240

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

    matrix<float> q;
    matrix<float> k;
    matrix<float> attn_weights;
    matrix<float> x1;

    matrix<float> dequant;

    TransformerBuffer(int seqlen, const TransformerConfig &cfg, int dequant_size = 1):
        seqlen(seqlen),
        cfg(cfg),
        q(cfg.num_heads * seqlen, cfg.head_dim()),
        k(cfg.num_heads * seqlen, cfg.head_dim()),
        attn_weights(cfg.num_heads * seqlen, seqlen),
        x1(seqlen, cfg.ffn_embed_dim),
        dequant(dequant_size, 1)
    {}
};


void smart_load_(matrix<float> & A, const std::string & path) {
    load_(A, path);
}

template <int block_size>
void smart_load_(quantized_matrix<Q4, block_size> & A, const std::string & path) {
    std::string path_trunk = path.substr(0, path.size() - 4);
    std::string metadata_path = path_trunk + QMATRIX_META_SUFFIX;

    std::ifstream qmeta_file(metadata_path);
    if (qmeta_file.good()) {
        // If quantization metadata file exists, directly load it
        load_(A, metadata_path);
    } else {
        // Otherwise, load the full-precision matrix and quantize it
        matrix<float> *temp = new matrix<float>(A.n_rows, A.n_cols);
        load_(*temp, path);
        quantize(*temp, &A);
        delete temp;
    }
}

template <typename WeightType=matrix<float>>
struct TransformerLayer {
    TransformerConfig cfg;
    matrix<float> self_attn_layer_norm_weight;
    matrix<float> self_attn_layer_norm_bias;

    WeightType k_proj_weight;
    matrix<float> k_proj_bias;
    WeightType v_proj_weight;
    matrix<float> v_proj_bias;
    WeightType q_proj_weight;
    matrix<float> q_proj_bias;
    WeightType out_proj_weight;
    matrix<float> out_proj_bias;

    WeightType fc1_weight;
    matrix<float> fc1_bias;
    WeightType fc2_weight;
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
        final_layer_norm_weight(cfg.embed_dim, 1),
        final_layer_norm_bias(cfg.embed_dim, 1),
        fc1_weight(cfg.ffn_embed_dim, cfg.embed_dim),
        fc1_bias(cfg.ffn_embed_dim, 1),
        fc2_weight(cfg.embed_dim, cfg.ffn_embed_dim),
        fc2_bias(cfg.embed_dim, 1)
    {
        load_(self_attn_layer_norm_weight, dirpath + "/self_attn_layer_norm.weight.bin");
        load_(self_attn_layer_norm_bias, dirpath + "/self_attn_layer_norm.bias.bin");
        smart_load_(k_proj_weight, dirpath + "/self_attn.k_proj.weight.bin");
        load_(k_proj_bias, dirpath + "/self_attn.k_proj.bias.bin");
        smart_load_(v_proj_weight, dirpath + "/self_attn.v_proj.weight.bin");
        load_(v_proj_bias, dirpath + "/self_attn.v_proj.bias.bin");
        smart_load_(q_proj_weight, dirpath + "/self_attn.q_proj.weight.bin");
        load_(q_proj_bias, dirpath + "/self_attn.q_proj.bias.bin");
        mul_(q_proj_weight, 1.0f / sqrtf(cfg.head_dim()));
        mul_(q_proj_bias, 1.0f / sqrtf(cfg.head_dim()));

        smart_load_(out_proj_weight, dirpath + "/self_attn.out_proj.weight.bin");
        load_(out_proj_bias, dirpath + "/self_attn.out_proj.bias.bin");
        smart_load_(fc1_weight, dirpath + "/fc1.weight.bin");
        load_(fc1_bias, dirpath + "/fc1.bias.bin");
        smart_load_(fc2_weight, dirpath + "/fc2.weight.bin");
        load_(fc2_bias, dirpath + "/fc2.bias.bin");
        load_(final_layer_norm_weight, dirpath + "/final_layer_norm.weight.bin");
        load_(final_layer_norm_bias, dirpath + "/final_layer_norm.bias.bin");
    }

    TransformerBuffer * create_buffer(int seqlen) const {
        int dq_size = 1;
        dq_size = std::max(dq_size, k_proj_weight.numel());
        dq_size = std::max(dq_size, v_proj_weight.numel());
        dq_size = std::max(dq_size, q_proj_weight.numel());
        dq_size = std::max(dq_size, out_proj_weight.numel());
        dq_size = std::max(dq_size, fc1_weight.numel());
        dq_size = std::max(dq_size, fc2_weight.numel());

        return new TransformerBuffer(seqlen, cfg, dq_size);
    }

    void operator() (const matrix<float> &x, matrix<float> &y, TransformerBuffer &buffer) {
        START_TIMER();

        layer_norm(x, self_attn_layer_norm_weight, self_attn_layer_norm_bias, y); RECORD_TIME("layer_norm");
        attn_proj_linear(y, q_proj_weight, q_proj_bias, buffer.q, &buffer.dequant); RECORD_TIME("q_proj_linear");
        attn_proj_linear(y, k_proj_weight, k_proj_bias, buffer.k, &buffer.dequant); RECORD_TIME("k_proj_linear");
        apply_rotary_embedding_(buffer.q, cfg.num_heads); RECORD_TIME("apply_rotary_embedding");
        apply_rotary_embedding_(buffer.k, cfg.num_heads); RECORD_TIME("apply_rotary_embedding");
        bmm<true>(buffer.q, buffer.k, buffer.attn_weights); RECORD_TIME("bmm");
        softmax_(buffer.attn_weights); RECORD_TIME("softmax");

        attn_proj_linear(y, v_proj_weight, v_proj_bias, buffer.k, &buffer.dequant);  // Use k buffer to save memory
        RECORD_TIME("v_proj_linear");
        bmm<false>(buffer.attn_weights, buffer.k, buffer.q);  // Use q buffer to save memory
        RECORD_TIME("bmm");
        
        attn_out_linear(buffer.q, out_proj_weight, out_proj_bias, y, &buffer.dequant); RECORD_TIME("attn_out_linear");
        add_(y, x); RECORD_TIME("add");

        fused_layer_norm_linear_gelu(y, final_layer_norm_weight, final_layer_norm_bias, fc1_weight, fc1_bias, buffer.x1, &buffer.dequant); RECORD_TIME("fused_layer_norm_linear");
        output_linear_residual(buffer.x1, fc2_weight, fc2_bias, y, &buffer.dequant); RECORD_TIME("linear_residual");
    }
};

typedef matrix<float> Weight_FP32;
typedef quantized_matrix<Q4, 128> Weight_Q4;

template <typename WeightType>
TransformerLayer<WeightType> * load_transformer_layer(const std::string &dirpath) {
    TransformerConfig cfg(dirpath + "/config.txt");
    return new TransformerLayer<WeightType>(cfg, dirpath);
}

#endif // PLM_TRANSFORMER_H