#ifndef PLM_ESM_H
#define PLM_ESM_H

#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <cmath>
#include "../matrix.h"
#include "../kernels.h"
#include "transformer.h"
#include "alphabet.h"
#include "transformer.h"

struct ESMConfig {
    int num_layers;
    int embed_dim;
    int attention_heads;
    TransformerConfig transformer_cfg;

    void check_config() const {
        if (transformer_cfg.embed_dim != embed_dim) {
            std::cerr << "Mismatched embed_dim in ESMConfig and TransformerConfig" << std::endl;
            exit(1);
        }
        if (transformer_cfg.num_heads != attention_heads) {
            std::cerr << "Mismatched attention_heads in ESMConfig and TransformerConfig" << std::endl;
            exit(1);
        }
    }

    ESMConfig(int num_layers, int embed_dim, int attention_heads, const TransformerConfig &transformer_cfg):
        num_layers(num_layers),
        embed_dim(embed_dim),
        attention_heads(attention_heads),
        transformer_cfg(transformer_cfg)
    {
        check_config();
    }
};

struct ESMBuffer {
    int total_length;
    ESMConfig cfg;
    std::unique_ptr<TransformerBuffer> transformer_buffer;

    matrix<float> x;
    matrix<float> y;

    ESMBuffer(int total_length, const ESMConfig &cfg, TransformerBuffer *transformer_buffer):
        total_length(total_length),
        cfg(cfg),
        transformer_buffer(transformer_buffer),
        x(total_length, cfg.embed_dim),
        y(total_length, cfg.embed_dim)
    {}
};


struct ESMRepresentation {
    int seqlen;
    ESMConfig cfg;
    matrix<float> esm_s_combine_normalized;
    matrix<float> s;
    matrix<float> z;

    const bool memory_saving_mode;
    matrix<float> z_moment1;
    matrix<float> z_moment2;
    matrix<float> z_layernorm_weight;
    matrix<float> z_layernorm_bias;
    matrix<float> z_linear_weight;
    matrix<float> z_linear_bias;

    bool is_finalized = false;

    ESMRepresentation(int seqlen, const ESMConfig &cfg, const matrix<float> &esm_s_combine_normalized):
        seqlen(seqlen),
        cfg(cfg),
        esm_s_combine_normalized(esm_s_combine_normalized),
        s(seqlen, cfg.embed_dim),
        z(seqlen * seqlen, cfg.num_layers * cfg.attention_heads),
        memory_saving_mode(false),
        z_moment1(0, 0),
        z_moment2(0, 0),
        z_layernorm_weight(0, 0),
        z_layernorm_bias(0, 0),
        z_linear_weight(0, 0),
        z_linear_bias(0, 0)
    {}

    ESMRepresentation(
        int seqlen,
        const ESMConfig &cfg,
        const matrix<float> &esm_s_combine_normalized,
        const matrix<float> &z_layernorm_weight,
        const matrix<float> &z_layernorm_bias,
        const matrix<float> &z_linear_weight,
        const matrix<float> &z_linear_bias):
        seqlen(seqlen),
        cfg(cfg),
        esm_s_combine_normalized(esm_s_combine_normalized),
        s(seqlen, cfg.embed_dim),
        z(seqlen * seqlen, z_linear_weight.n_rows),
        memory_saving_mode(true),
        z_moment1(seqlen * seqlen, 1),
        z_moment2(seqlen * seqlen, 1),
        z_layernorm_weight(z_layernorm_weight),
        z_layernorm_bias(z_layernorm_bias),
        z_linear_weight(z_linear_weight),
        z_linear_bias(z_linear_bias)
    {}

    void accumulate_s(const matrix<float> &s_esm, int layer) {
        // s_esm:  (1 + seqlen + 1, embed_dim), with sos and eos token embeddings
        for (int i = 0; i < seqlen; i ++) {
            for (int j = 0; j < cfg.embed_dim; j ++) {
                *s(i, j) += *s_esm(i + 1, j) * *esm_s_combine_normalized(layer, 0);
            }
        }
    }

    void _finalize(int num_layers) {
        if (is_finalized) {
            std::cerr << "Already finalized" << std::endl;
            exit(1);
        }
        is_finalized = true;
        if (memory_saving_mode) {
            int c_z = z.n_cols;
            int in_channels = cfg.attention_heads * num_layers;
            #pragma omp parallel for
            for (int i = 0; i < seqlen; i ++) {
                for (int j = 0; j < seqlen; j ++) {
                    int z_loc = i * seqlen + j;
                    float mean = *z_moment1(z_loc, 0) / in_channels;
                    float var = *z_moment2(z_loc, 0) / in_channels - mean * mean;
                    float inv_std = 1.0f / std::sqrt(var + 1e-5);
                    float mean_div_std = mean * inv_std;
                    for (int k = 0; k < c_z; k ++) {
                        float Bl = *z_linear_bias(k, 0);
                        float WnWl = 0.0;
                        float BnWl = 0.0;
                        for (int l = 0; l < z_layernorm_weight.n_rows; l ++) {
                            WnWl += *z_layernorm_weight(l, 0) * *z_linear_weight(k, l);
                            BnWl += *z_layernorm_bias(l, 0) * *z_linear_weight(k, l);
                        }
                        float z_original = *z(z_loc, k);
                        *z(z_loc, k) = gelu_scalar(z_original * inv_std - mean_div_std * WnWl + BnWl + Bl);
                    }
                }
            }
        }
    }

    void save_z(const matrix<float> &attn_weights, int layer) {
        // attn_weights: (num_heads * 1+seqlen+1, 1+seqlen+1)
        if (is_finalized) {
            std::cerr << "ESM Representation has been finalized. Please create a new one to accumulate new results." << std::endl;
            exit(1);
        }
        if (memory_saving_mode) {
            int c_z = z.n_cols;
            int in_channel_start = layer * cfg.attention_heads;
            #pragma omp parallel for
            for (int i = 0; i < seqlen; i ++) {
                for (int j = 0; j < seqlen; j ++) {
                    int z_loc = j * seqlen + i;
                    for (int k = 0; k < cfg.attention_heads; k ++) {
                        int attn_row = k * (seqlen + 2) + i + 1;
                        int attn_col = j + 1;
                        float aw = *attn_weights(attn_row, attn_col);
                        *z_moment1(z_loc, 0) += aw;
                        *z_moment2(z_loc, 0) += aw * aw;
                    }

                    for (int l = 0; l < c_z; l ++) {
                        float sum = 0.0f;
                        for (int k = 0; k < cfg.attention_heads; k ++) {
                            int attn_row = k * (seqlen + 2) + i + 1;
                            int attn_col = j + 1;
                            float attn = *attn_weights(attn_row, attn_col);
                            float w_layernorm = *z_layernorm_weight(in_channel_start + k, 0);
                            float w_linear = *z_linear_weight(l, in_channel_start + k);
                            sum += attn * w_layernorm * w_linear;
                        }
                        *z(z_loc, l) += sum;
                    }
                }
            }
            if (layer == cfg.num_layers - 1) {
                _finalize(cfg.num_layers);
            }
        } else {
            for (int i = 0; i < seqlen; i ++) {
                for (int j = 0; j < seqlen; j ++) {
                    for (int k = 0; k < cfg.attention_heads; k ++) {
                        // NOTE: The order here is (j, i) according to esmfold.py L140
                        *z(j * seqlen + i, layer * cfg.attention_heads + k) = *attn_weights(k * (seqlen + 2) + i + 1, j + 1);
                    }
                }
            }
        }
    }
};

template <typename TransformerWeightType>
struct ESM {
    ESMConfig cfg;
    matrix<float> embed_tokens;
    std::vector<std::unique_ptr<TransformerLayer<TransformerWeightType>>> transformer_layers;
    matrix<float> emb_layer_norm_after_weight;
    matrix<float> emb_layer_norm_after_bias;

    ESM(const ESMConfig &cfg, const std::string &dirpath, std::vector<TransformerLayer<TransformerWeightType> *> transformer_layer_ptrs):
        cfg(cfg),
        embed_tokens(ALPHABET_SIZE, cfg.embed_dim),
        emb_layer_norm_after_weight(cfg.embed_dim, 1),
        emb_layer_norm_after_bias(cfg.embed_dim, 1)
    {
        std::cerr << "Loading weights for ESM from " << dirpath << std::endl;
        load_(embed_tokens, dirpath + "/embed_tokens.weight.bin");
        load_(emb_layer_norm_after_weight, dirpath + "/emb_layer_norm_after.weight.bin");
        load_(emb_layer_norm_after_bias, dirpath + "/emb_layer_norm_after.bias.bin");
        for (int i = 0; i < cfg.num_layers; i ++) {
            transformer_layers.emplace_back(transformer_layer_ptrs[i]);
        }
        if (transformer_layers.size() != cfg.num_layers) {
            std::cerr << "Mismatched number of transformer layers" << std::endl;
            exit(1);
        }
        if (transformer_layers.size() == 0) {
            std::cerr << "No transformer layers" << std::endl;
            exit(1);
        }

        std::vector<int> awq_enabled_layers;
        for (int i = 0; i < cfg.num_layers; i ++) {
            if (transformer_layers[i]->awq_state == AWQEnabled) {
                awq_enabled_layers.push_back(i);
            }
        }
        if (awq_enabled_layers.size() > 0) {
            std::cerr << "AWQ is enabled for layer: ";
            for (int i = 0; i < awq_enabled_layers.size(); i ++) {
                std::cerr << awq_enabled_layers[i] << " ";
            }
            std::cerr << std::endl;
        }
    }

    ESMBuffer * create_buffer(const matrix<int> &esm_aatype) {
        int seqlen = esm_aatype.n_rows;
        return new ESMBuffer(seqlen + 2, cfg, transformer_layers[0]->create_buffer(seqlen + 2));
    }

    void operator() (const matrix<int> &esm_aatype, ESMBuffer &buffer, ESMRepresentation *repr_out = nullptr, int stop_at = -1) const {
        stop_at = stop_at < 0 ? cfg.num_layers : stop_at;

        for (int i = 0; i < cfg.embed_dim; i ++) {
            *buffer.x(0, i) = *embed_tokens(CLS_TOKEN, i);
        }
        for (int i = 0; i < buffer.total_length - 2; i ++) {
            for (int j = 0; j < cfg.embed_dim; j ++) {
                int aat = *esm_aatype(i, 0);
                if (aat < 0 || aat >= ALPHABET_SIZE) {
                    std::cerr << "Invalid amino acid type" << std::endl;
                    aat = UNK_TOKEN;
                }
                *buffer.x(i + 1, j) = *embed_tokens(aat, j);
            }
        }
        for (int i = 0; i < cfg.embed_dim; i ++) {
            *buffer.x(buffer.total_length - 1, i) = *embed_tokens(EOS_TOKEN, i);
        }

        if (repr_out != nullptr) {
            repr_out->accumulate_s(buffer.x, 0);
        }

        if (stop_at == 0) {
            return;
        }

        matrix<float> *in_ptr, *out_ptr;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < std::min(cfg.num_layers, stop_at); i ++) {
            std::cerr << "Running transformer layer #" << i + 1 << "/" << cfg.num_layers << " ...";
            if (i % 2 == 0) {
                in_ptr = &buffer.x;
                out_ptr = &buffer.y;
            } else {
                in_ptr = &buffer.y;
                out_ptr = &buffer.x;
            }
            (*transformer_layers[i])(*in_ptr, *out_ptr, *buffer.transformer_buffer);
            if (i == cfg.num_layers - 1) {
                layer_norm(*out_ptr, emb_layer_norm_after_weight, emb_layer_norm_after_bias, *out_ptr);
            }
            auto end = std::chrono::steady_clock::now();
            // Report in seconds
            int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cerr << "(elapsed: " << elapsed_ms / 1000.0 << "s, average: " << (elapsed_ms / (i + 1)) / 1000.0 << "s)" << "\t\r" << std::flush;
            if (repr_out != nullptr) {
                repr_out->accumulate_s(*out_ptr, i + 1);
                repr_out->save_z(buffer.transformer_buffer->attn_weights, i);
            }
        }
        std::cerr << std::endl;
    }
};

std::string get_transformer_layer_path(const std::string &dirpath, int layer) {
    return dirpath + "/transformer_" + std::to_string(layer);
}


typedef ESM<Weight_FP32> ESM_fp32;


template <typename TransformerWeightType>
ESM<TransformerWeightType>* load_esm(const std::string &dirpath) {
    int num_layers, embed_dim, attention_heads;
    std::ifstream cfg_file(dirpath + "/config.txt");
    if (!cfg_file.is_open())
    {
        std::cerr << "Could not open config file" << std::endl;
        exit(1);
    }
    cfg_file >> num_layers >> embed_dim >> attention_heads;

    std::vector<TransformerLayer<TransformerWeightType> *> transformer_layers;
    for (int i = 0; i < num_layers; i ++) {
        transformer_layers.push_back(nullptr);
    }

    int count_loaded = 0;
    #pragma omp parallel for
    for (int i = 0; i < num_layers; i ++) {
        auto *ptr = load_transformer_layer<TransformerWeightType>(get_transformer_layer_path(dirpath, i));
        #pragma omp critical
        {
            transformer_layers[i] = ptr;
            count_loaded += 1;
            std::cerr << "Loading transformer layer ..." << "(" << count_loaded << "/" << num_layers << ")\t\r" << std::flush;
        }
    }
    std::cerr << std::endl;
    ESMConfig esm_cfg(num_layers, embed_dim, attention_heads, transformer_layers[0]->cfg);
    return new ESM(esm_cfg, dirpath, transformer_layers);
}

#endif // PLM_ESM_H