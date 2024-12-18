import re
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from tinyfold.data.alphabet import Alphabet
from tinyfold.utils.export import export_tensor_dict, export_value_list

from .heads import ContactPredictionHead, RobertaLMHead
from .transformer import TransformerLayer


class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Alphabet | str = "ESM-1b",
        token_dropout: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        # self.token_dropout = token_dropout
        self.token_dropout = False

        self._init_submodules()

    @classmethod
    def load(self, path: str):
        def upgrade_state_dict(state_dict):
            """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
            prefixes = ["encoder.sentence_encoder.", "encoder."]
            pattern = re.compile("^" + "|".join(prefixes))
            state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
            return state_dict

        model_data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = model_data["cfg"]["model"]
        state_dict = model_data["model"]
        state_dict = upgrade_state_dict(state_dict)

        contact_path = path.replace(".pt", "-contact-regression.pt")
        contact_data = torch.load(contact_path, map_location="cpu", weights_only=False)
        state_dict.update(contact_data["model"])

        model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            alphabet="ESM-1b",
            token_dropout=cfg.token_dropout,
        )
        model.load_state_dict(state_dict)
        return model

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(tqdm(self.layers, desc="Transformer layer forward")):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    def export(self, dirpath: str | Path) -> None:
        dirpath = Path(dirpath)

        for i, layer in enumerate(tqdm(self.layers, desc="Exporting transformer layers")):
            layer.export(dirpath / f"transformer_{i}")

        sd = {k: v for k, v in self.state_dict().items() if not k.startswith("layers.")}
        export_tensor_dict(sd, dirpath)
        export_value_list(
            [
                self.num_layers,
                self.embed_dim,
                self.attention_heads,
            ],
            dirpath / "config.txt",
        )
