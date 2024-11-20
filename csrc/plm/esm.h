#ifndef PLM_ESM_H
#define PLM_ESM_H

#include <vector>
#include <memory>
#include <string>
#include "../matrix.h"
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
    ESMRepresentation(int seqlen, const ESMConfig &cfg, const matrix<float> &esm_s_combine_normalized):
        seqlen(seqlen),
        cfg(cfg),
        esm_s_combine_normalized(esm_s_combine_normalized),
        s(seqlen, cfg.embed_dim),
        z(seqlen * seqlen, cfg.num_layers * cfg.attention_heads)
    {}

    void accumulate_s(const matrix<float> &s_esm, int layer) {
        // s_esm:  (1 + seqlen + 1, embed_dim), with sos and eos token embeddings
        for (int i = 0; i < seqlen; i ++) {
            for (int j = 0; j < cfg.embed_dim; j ++) {
                *s(i, j) += *s_esm(i + 1, j) * *esm_s_combine_normalized(layer, 0);
            }
        }
    }

    void save_z(const matrix<float> &attn_weights, int layer) {
        // attn_weights: (num_heads * 1+seqlen+1, 1+seqlen+1)
        for (int i = 0; i < seqlen; i ++) {
            for (int j = 0; j < seqlen; j ++) {
                for (int k = 0; k < cfg.attention_heads; k ++) {
                    *z(i * seqlen + j, layer * cfg.attention_heads + k) = *attn_weights(k * (seqlen + 2) + i + 1, j + 1);
                }
            }
        }
    }
};

struct ESM {
    ESMConfig cfg;
    matrix<float> embed_tokens;
    std::vector<std::unique_ptr<TransformerLayer>> transformer_layers;
    matrix<float> emb_layer_norm_after_weight;
    matrix<float> emb_layer_norm_after_bias;

    ESM(const ESMConfig &cfg, const std::string &dirpath, std::vector<TransformerLayer *> transformer_layer_ptrs):
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
        for (int i = 0; i < std::min(cfg.num_layers, stop_at); i ++) {
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
            if (repr_out != nullptr) {
                repr_out->accumulate_s(*out_ptr, i + 1);
                repr_out->save_z(buffer.transformer_buffer->attn_weights, i);
            }
        }
    }
};

std::string get_transformer_layer_path(const std::string &dirpath, int layer) {
    return dirpath + "/transformer_" + std::to_string(layer);
}

ESM * load_esm(const std::string &dirpath) {
    int num_layers, embed_dim, attention_heads;
    std::ifstream cfg_file(dirpath + "/config.txt");
    if (!cfg_file.is_open())
    {
        std::cerr << "Could not open config file" << std::endl;
        exit(1);
    }
    cfg_file >> num_layers >> embed_dim >> attention_heads;

    std::vector<TransformerLayer *> transformer_layers;
    for (int i = 0; i < num_layers; i ++) {
        transformer_layers.push_back(load_transformer_layer(get_transformer_layer_path(dirpath, i)));
    }
    ESMConfig esm_cfg(num_layers, embed_dim, attention_heads, transformer_layers[0]->cfg);
    return new ESM(esm_cfg, dirpath, transformer_layers);
}

#endif // PLM_ESM_H