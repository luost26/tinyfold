import click
import torch
import numpy as np
import pathlib
from matplotlib import pyplot as plt
from matplotlib import cm
from tinyfold.utils.awq import get_calib_feat_and_weight, model_weight_auto_scale

from tinyfold.models.esmfold import ESMFold 

def visualize(name, status, weights, activations, zlim, output_dir):
    activations = activations.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(activations.shape[1])
    y = np.arange(activations.shape[0])
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, activations, cmap=cm.coolwarm, cstride=10, rstride=8, vmin=0, vmax=0.4, linewidth=0, antialiased=False)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Token")
    ax.set_zlim(0, zlim)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(output_dir / f"{name}.self_attn.qkv_proj.activation.{status}_AWQ.png")
    
    qkv = ["q", "k", "v"]
    for i, w in enumerate(weights):
        w = w.numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x = np.arange(w.shape[1])
        y = np.arange(w.shape[0])
        x, y = np.meshgrid(x, y)
        surf = ax.plot_surface(x, y, w, cmap=cm.coolwarm, cstride=10, rstride=8, vmin=0, vmax=0.4, linewidth=0, antialiased=False)
        ax.set_xlabel("Out Channel")
        ax.set_ylabel("In Channel")
        ax.set_zlim(0, zlim)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(output_dir / f"{name}.self_attn.{qkv[i]}_proj.weight.{status}_AWQ.png")
    
    print(f"Visualize weights and activations {status} AWQ in {output_dir}.")
    

@click.command()
@click.option("--device", default="cuda")
@click.option("--w_bit", default=4)
@click.option("--layer", default="esm.layers.1", 
              help="The exemple layer for visualizing qkv projection weights and activations before and after AWQ. Options: esm.layers.{1-35}")
@click.option("data_path", "--data", type=click.Path(exists=True), default="./data/testset/family_0.pkl")
@click.option("esm_path", "--esm", type=click.Path(exists=True), default="./data/esm2_t36_3B_UR50D.pt")
@click.option(
    "esmfold_path",
    "--esmfold",
    type=click.Path(exists=True),
    default="./data/esmfold_structure_module_only_3B.pt",
)
@click.option("output_dir", "--out", type=click.Path(path_type=pathlib.Path), default="./data/output/awq")
def main(esm_path: str, esmfold_path: str, data_path: str, output_dir: pathlib.Path, device: str, layer: str, w_bit: int):

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading model from {esmfold_path} and {esm_path}")
    model = ESMFold.load(esmfold_path, esm_path).to(device)            
    model.eval()

    torch.set_grad_enabled(False)
    input_dict, weight_dict = get_calib_feat_and_weight(model, data_path)
    weights = [weight_dict[layer + f".self_attn.{qkv}_proj"] for qkv in ["q", "k", "v"]]
    activations = torch.stack(input_dict[layer + ".self_attn.q_proj"])
    max_act = torch.max(activations).item()
    visualize(name=layer, status="before", weights=weights, activations=activations, zlim=max_act, output_dir=output_dir)
    
    best_scales = model_weight_auto_scale(model, input_feat=input_dict, w_bit=w_bit)
    
    for w in weights:
        w *= best_scales[layer]
    activations /= best_scales[layer]
    visualize(name=layer, status="after", weights=weights, activations=activations, zlim=max_act, output_dir=output_dir)

if __name__ == "__main__":
    main()
