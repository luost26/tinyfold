import torch
import random
import sys
from pathlib import Path

from efficient_esm.models.structure_module import InvariantPointAttention, Rigid, StructureModule
from efficient_esm.models.esmfold import ESMFold
from efficient_esm.models.openfold import residue_constants
from efficient_esm.models.transformer import TransformerLayer
from efficient_esm.utils.export import export_tensor_dict
from efficient_esm.utils.quantize import pseudo_quantize_tensor
from efficient_esm.models.esm2 import ESM2
from efficient_esm.models.esm_misc import batch_encode_sequences
from efficient_esm.data.alphabet import Alphabet

torch.set_grad_enabled(False)


def ipa():
    torch.manual_seed(0)
    random.seed(0)

    output_dir = Path("./data/c_test/ipa")

    seqlen = 17
    module = InvariantPointAttention(32, 8, 16, 4, 2, 2)
    module.eval()
    module.export(output_dir)

    s = torch.randn([1, seqlen, 32])
    z = torch.randn([1, seqlen, seqlen, 8])
    quat = torch.randn([1, seqlen, 4])
    r = Rigid.from_tensor_7(torch.randn([1, seqlen, 7]), True)
    mask = torch.ones([1, seqlen])

    out, v = module(s, z, r, mask, return_intermediates=True)

    export_tensor_dict({"s": s, "z": z, "r": r.to_tensor_4x4()}, output_dir / "input")
    export_tensor_dict(v, output_dir / "output")
    print(f"IPA test data generated: {output_dir}")


def structure_module():
    torch.manual_seed(0)
    random.seed(0)

    seqlen = 17
    cfg = {
        "c_s": 32,
        "c_z": 16,
        "c_ipa": 8,
        "c_resnet": 12,
        "no_heads_ipa": 12,
        "no_qk_points": 4,
        "no_v_points": 8,
        "dropout_rate": 0.1,
        "no_blocks": 8,
        "no_transition_layers": 1,
        "no_resnet_blocks": 2,
        "no_angles": 7,
        "trans_scale_factor": 10,
        "epsilon": 1e-08,
        "inf": 100000.0,
    }

    output_dir = Path("./data/c_test/structure_module")
    module = StructureModule(**cfg)
    module.eval()

    single = torch.randn([1, seqlen, int(cfg["c_s"])])
    pair = torch.randn([1, seqlen, seqlen, int(cfg["c_z"])])
    aatype = torch.randint(0, 20, [1, seqlen])
    module.export(output_dir)

    out = module({"single": single, "pair": pair}, aatype, return_intermediates=True)
    export_tensor_dict({"s": single, "z": pair, "aatype": aatype}, output_dir / "input")
    export_tensor_dict(out, output_dir / "output")


def esmfold_folding_only():
    esmfold_path = "./data/esmfold_structure_module_only_3B.pt"
    esm_path = "./data/esm2_t36_3B_UR50D.pt"
    device = "cuda"
    print(f"Loading model from {esmfold_path} and {esm_path}")
    model = ESMFold.load(esmfold_path, esm_path).to(device)
    model.eval()
    print("Exporting model")
    model.export_folding(Path("./data/c_test/esmfold_folding_only"))

    print("Generating test data")
    seq = "ASAWPEEKNYHQPAILNSSALRQIAEGTSISEMWQNDLQPLLIERYPGSPGSYAARQHIMQRIQRLQADWVLEIDTFLSQTPYGYRSFSNIISTLNPTAKRHLVLACHYDSKYFSHWNNRVFVGATDS"  # noqa
    out = model.infer(seq)
    with open("./data/c_test/esmfold_folding_only/out.pdb", "w") as f:
        f.writelines(model.output_to_pdb(out))
    plddt = out["mean_plddt"].item()
    print(f"pLDDT: {plddt:.2f}")

    export_tensor_dict(
        {"esm_s": out["esm_s"], "esm_z": out["esm_z"], "aatype": out["aatype"], "residx": out["residx"]},
        Path("./data/c_test/esmfold_folding_only") / "input",
    )
    export_tensor_dict(out, Path("./data/c_test/esmfold_folding_only") / "output")


def transformer():
    torch.manual_seed(0)
    random.seed(0)
    seqlen = 17

    output_dir = Path("./data/c_test/transformer")
    module = TransformerLayer(32, 64, 4, True)
    module.eval()
    x = torch.randn([1, seqlen, 32])
    y, attn_map, intermediates = module(x.transpose(0, 1).contiguous(), return_intermediates=True)

    module.export(output_dir)
    export_tensor_dict({"x": x}, output_dir / "input")
    outs = {"out": y, "out_attn_map": attn_map, **intermediates}
    export_tensor_dict(outs, output_dir / "output")
    for k, v in outs.items():
        print(f"{k}: {v.shape}")


def esm_small():
    torch.manual_seed(0)
    random.seed(0)
    seqlen = 17
    seq = [0] + [random.randint(4, 24) for _ in range(seqlen)] + [2]

    output_dir = Path("./data/c_test/esm_small")
    module = ESM2(num_layers=4, embed_dim=128, attention_heads=8, token_dropout=False)
    module.eval()
    x = torch.tensor(seq).unsqueeze(0)
    result = module(x, repr_layers=[0, 1, 2, 3, 4], need_head_weights=True)
    module.export(output_dir)
    export_tensor_dict({"tokens": x[:, 1:-1]}, output_dir / "input")

    out_dict: dict[str, torch.Tensor] = {}
    out_dict["attentions"] = result["attentions"][0].permute(2, 3, 0, 1)
    out_dict["attentions_seq"] = result["attentions"][0, :, :, 1:-1, 1:-1].permute(2, 3, 0, 1)
    for k, v in result["representations"].items():
        out_dict[f"representations_{k}"] = v[0]
    export_tensor_dict(out_dict, output_dir / "output")

    for k, v in out_dict.items():
        print(f"{k}: {v.shape}")


def esm_full_3B():
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [d.get_idx(v) for v in residue_constants.restypes_with_x]
        return torch.tensor(esm_reorder)

    torch.manual_seed(0)
    random.seed(0)
    seq = "ASAWPEEKNYHQPAILNSSALRQIAEGTSISEMWQNDLQPLLIERYPGSPGSYAARQHIMQRIQRLQADWVLEIDTFLSQTPYGYRSFSNIISTLNPTAKRHLVLACHYDSKYFSHWNNRVFVGATDS"  # noqa

    output_dir = Path("./data/c_test/esm_full_3B")
    module = ESM2.load("data/esm2_t36_3B_UR50D.pt").to("cuda")
    module.eval()

    af2_to_esm = _af2_to_esm(module.alphabet)
    aatype, *_ = batch_encode_sequences([seq])
    esmaa = af2_to_esm[aatype + 1]
    batch_size = 1
    bosi, eosi = module.alphabet.cls_idx, module.alphabet.eos_idx
    bos = esmaa.new_full((batch_size, 1), bosi)
    eos = esmaa.new_full((batch_size, 1), module.alphabet.padding_idx)
    esmaa = torch.cat([bos, esmaa, eos], dim=1)
    # Use the first padding index as eos during inference.
    esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

    esmfold_ckpt = torch.load("./data/esmfold_structure_module_only_3B.pt", map_location="cpu")
    esm_s_combine = esmfold_ckpt["model"]["esm_s_combine"].to("cuda")

    result = module(esmaa.to("cuda"), repr_layers=range(module.num_layers + 1), need_head_weights=True)
    esm_s = torch.stack([v for _, v in sorted(result["representations"].items())], dim=2)
    esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
    esm_s = (esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)

    export_tensor_dict(
        {
            "tokens": esmaa[:, 1:-1],
            "esm_s_combine_normalized": esm_s_combine.softmax(0),
        },
        output_dir / "input",
    )

    out_dict: dict[str, torch.Tensor] = {
        "esm_s": esm_s,
    }
    out_dict["attentions"] = result["attentions"][0].permute(3, 2, 0, 1)  # 3,2 is consistent to esmfold.py L140
    out_dict["attentions_seq"] = result["attentions"][0, :, :, 1:-1, 1:-1].permute(3, 2, 0, 1)
    for k, v in result["representations"].items():
        out_dict[f"representations_{k}"] = v[0]
        out_dict[f"representations_seq_{k}"] = v[0, 1:-1]
    export_tensor_dict(out_dict, output_dir / "output")
    module.export(output_dir)

    for k, v in out_dict.items():
        print(f"{k}: {v.shape}")


def esmfold():
    esmfold_path = "./data/esmfold_structure_module_only_3B.pt"
    esm_path = "./data/esm2_t36_3B_UR50D.pt"
    device = "cuda"
    dirpath = Path("./data/c_test/esmfold")
    print(f"Loading model from {esmfold_path} and {esm_path}")
    model = ESMFold.load(esmfold_path, esm_path).to(device)
    model.eval()
    print("Exporting model")
    model.export(dirpath)

    print("Generating test data")
    seq = "ASAWPEEKNYHQPAILNSSALRQIAEGTSISEMWQNDLQPLLIERYPGSPGSYAARQHIMQRIQRLQADWVLEIDTFLSQTPYGYRSFSNIISTLNPTAKRHLVLACHYDSKYFSHWNNRVFVGATDS"  # noqa
    out = model.infer(seq)
    with open(dirpath / "out.pdb", "w") as f:
        f.writelines(model.output_to_pdb(out))
    with open(dirpath / "seq.txt", "w") as f:
        f.write(seq + "\n")
    plddt = out["mean_plddt"].item()
    print(f"pLDDT: {plddt:.2f}")

    export_tensor_dict(
        {"esm_s": out["esm_s"], "esm_z": out["esm_z"], "aatype": out["aatype"], "residx": out["residx"]},
        dirpath / "input",
    )
    export_tensor_dict(out, dirpath / "output")


def pseudo_quantize():
    torch.manual_seed(0)
    random.seed(0)
    mat = torch.randn([128, 256])

    quantized = pseudo_quantize_tensor(mat, 4, [128])

    export_tensor_dict({"input": mat, "quantized": quantized}, Path("./data/c_test/pseudo_quantize"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_test_data.py <which>")
        sys.exit(1)
    which = sys.argv[1]
    globals()[which]()
