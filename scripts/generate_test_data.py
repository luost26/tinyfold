import torch
import random
from pathlib import Path
from efficient_esm.models.structure_module import InvariantPointAttention, Rigid, StructureModule
from efficient_esm.utils.export import export_tensor_dict

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


if __name__ == "__main__":
    structure_module()