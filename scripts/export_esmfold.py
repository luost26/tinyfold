import click
import pathlib
import shutil
import copy
import torch
from tinyfold.models.transformer import TransformerLayer
from tinyfold.models.esmfold import ESMFold
from tinyfold.utils.awq import get_calib_feat_and_weight, model_weight_auto_scale

@click.command()
@click.option("--device", default="cuda")
@click.option("data_path", "--data", type=click.Path(exists=True), default="./data/testset/family_0.pkl")
@click.option("esm_path", "--esm", type=click.Path(exists=True), default="./data/esm2_t36_3B_UR50D.pt")
@click.option(
    "esmfold_path",
    "--esmfold",
    type=click.Path(exists=True),
    default="./data/esmfold_structure_module_only_3B.pt",
)
@click.option("output_dir", "--out", type=click.Path(path_type=pathlib.Path), default="./data/esmfold_fp32")
@click.option("--awq", is_flag=True, default=False, help="Export ESMfold with AWQ")
@click.option("--w_bit", default=4)

def main(esm_path: str, esmfold_path: str, data_path: str, output_dir: pathlib.Path, awq: bool, w_bit: int, device: str):
    if output_dir.exists():
        click.confirm(f"{output_dir} already exists. Overwrite?", abort=True)
        shutil.rmtree(output_dir)
    print(f"Loading model from {esmfold_path} and {esm_path}")
    model = ESMFold.load(esmfold_path, esm_path)
    
    if awq:
        calib_model = copy.deepcopy(model).to(device)
        calib_model.eval()
        torch.set_grad_enabled(False)
        input_dict, _ = get_calib_feat_and_weight(calib_model, data_path)
        best_scales = model_weight_auto_scale(calib_model, input_feat=input_dict, w_bit=w_bit)
        awq_layers = best_scales.keys()
        for name, module in model.named_modules():
            if name in awq_layers:
                module.self_attn_qkv_proj_awq_scale = best_scales[name]
                     
    print("Exporting model")
    model.export(output_dir)


if __name__ == "__main__":
    main()
