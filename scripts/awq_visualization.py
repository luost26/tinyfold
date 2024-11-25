import pickle
import click
import torch
from torch import nn
import numpy as np
import pathlib
from matplotlib import pyplot as plt
from matplotlib import cm
from functools import partial

from tinyfold.models.esmfold import ESMFold


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
@click.option("output_dir", "--out", type=click.Path(path_type=pathlib.Path), default="./data/output")

def main(esm_path: str, esmfold_path: str, data_path: str, device: str, output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {esmfold_path} and {esm_path}")
    model = ESMFold.load(esmfold_path, esm_path).to(device)            
    model.eval()
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
            # only do awq for kqv projection in all transformer layers except the first one
            if any(kqv in name for kqv in [".k", ".q", ".v"]) and ".0" not in name:
                hooks.append(
                    m.register_forward_hook(
                        partial(stat_input_max_hook, name=name)))
            
    print("Collecting activation scales...")
    for data in dataset:
        seq = data['seq']
        model.infer(seq)
    for hook in hooks:
        hook.remove()
            
    output_dir = output_dir / "activation"
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in input_dict.items():
        activations = torch.stack(value).numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x = np.arange(activations.shape[1])
        y = np.arange(activations.shape[0])
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, activations, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Token")
        plt.savefig(output_dir / f"{key}.png")

if __name__ == "__main__":
    main()
