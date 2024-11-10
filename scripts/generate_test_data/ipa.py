import torch
import random
from pathlib import Path
from efficient_esm.models.structure_module import InvariantPointAttention, Rigid
from efficient_esm.utils.export import export_tensor_dict

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    torch.set_grad_enabled(False)

    output_dir = Path("./data/c_test/ipa")

    module = InvariantPointAttention(32, 8, 16, 4, 2, 2)
    module.eval()
    module.export(output_dir)

    seqlen = 17
    s = torch.randn([1, seqlen, 32])
    z = torch.randn([1, seqlen, seqlen, 8])
    quat = torch.randn([1, seqlen, 4])
    r = Rigid.from_tensor_7(torch.randn([1, seqlen, 7]), True)
    mask = torch.ones([1, seqlen])

    out, v = module(s, z, r, mask, return_intermediates=True)

    export_tensor_dict({"s": s, "z": z, "r": r.to_tensor_4x4()}, output_dir / "input")
    export_tensor_dict(v, output_dir / "output")
    print(f"IPA test data generated: {output_dir}")
