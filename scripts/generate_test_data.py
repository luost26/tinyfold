import torch
import random
import sys
from pathlib import Path
from efficient_esm.models.structure_module import InvariantPointAttention, Rigid, StructureModule
from efficient_esm.models.esmfold import ESMFold
from efficient_esm.models.transformer import TransformerLayer
from efficient_esm.utils.export import export_tensor_dict
from efficient_esm.models.esm2 import ESM2

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
    model.export(Path("./data/c_test/esmfold_folding_only"))

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
    seq = [1] + [random.randint(4, 24) for _ in range(seqlen - 2)] + [2]

    output_dir = Path("./data/c_test/esm_small")
    module = ESM2(num_layers=4, embed_dim=128, attention_heads=4)
    x = torch.tensor(seq).unsqueeze(0)
    result = module(x, repr_layers=[0, 1, 2, 3, 4], need_head_weights=True)
    print(result)
    module.export(output_dir)
    export_tensor_dict({"tokens": x}, output_dir / "input")
    export_tensor_dict(result["representations"], output_dir / "output")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_test_data.py <which>")
        sys.exit(1)
    which = sys.argv[1]
    globals()[which]()
