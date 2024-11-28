import click
import pathlib
import shutil
import torch
from torch import nn
import pickle
from functools import partial
from tinyfold.models.esmfold import ESMFold
import awq_visualization as awq

@click.command()
@click.option("data_path", "--data", type=click.Path(exists=True), default="./data/testset/family_0.pkl")
@click.option("esm_path", "--esm", type=click.Path(exists=True), default="./data/esm2_t36_3B_UR50D.pt")
@click.option(
    "esmfold_path",
    "--esmfold",
    type=click.Path(exists=True),
    default="./data/esmfold_structure_module_only_3B.pt",
)
@click.option("output_dir", "--out", type=click.Path(path_type=pathlib.Path), default="./data/esmfold_fp32")

def get_calib_feat(data_path: str, model: ESMFold):
    assert data_path.exists(), f"Calibration data {data_path} doesn't exist. Please generate one with ./scripts/create_testset.py."
    dataset = pickle.load(open(data_path, 'rb'))
    
    torch.set_grad_enabled(False)
    input_dict = dict()
    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]
            
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            # only do AWQ for qkv projection in all transformer layers except the first one
            if any(qkv in name for qkv in [".q", ".k", ".v"]) and ".0" not in name:
                hooks.append(
                    m.register_forward_hook(
                        partial(stat_input_max_hook, name=name)))

    print("Collecting activation scales...")    
    for data in dataset:
        seq = data['seq']
        model.infer(seq)
    for hook in hooks:
        hook.remove()
        
    return input_dict

def main(esm_path: str, esmfold_path: str, output_dir: pathlib.Path):
    if output_dir.exists():
        click.confirm(f"{output_dir} already exists. Overwrite?", abort=True)
        shutil.rmtree(output_dir)
    print(f"Loading model from {esmfold_path} and {esm_path}")
    model = ESMFold.load(esmfold_path, esm_path)
    print("Exporting model")
    model.export(output_dir)


if __name__ == "__main__":
    main()
