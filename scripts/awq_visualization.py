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


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
    max_val = w.amax(dim=1, keepdim=True)
    assert max_val.dim() == 2 and max_val.size(0) == w.size(0) and max_val.size(1) == 1
    min_val = w.amin(dim=1, keepdim=True)
    assert min_val.dim() == 2 and min_val.size(0) == w.size(0) and min_val.size(1) == 1

    # Calculate the scale factor and zero point.  (Formula 1 & 2)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    assert scales.shape == max_val.shape
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    assert w.dim() == 2 and w.size(0) == scales.size(0)
    if q_group_size > 0:
        assert w.size(1) == q_group_size

    # Dequantize W (pseudo quantization, the inverse transformation of Formula 3)
    w = (w - zeros) * scales
    assert w.dim() == 2 and w.size(0) == scales.size(0)
    if q_group_size > 0:
        assert w.size(1) == q_group_size

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w
            
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

def auto_scale_block(module, name, input_feat, w_bit, q_group_size):

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):

        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        s_x = x.view(-1, x.shape[-1]).abs().mean(0)

        # Initialize the best_error, best_ratio and best_scales
        best_error = float('inf')
        best_ratio = None
        best_scales = None

        n_grid = 80
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        
        # Calculate the quantization errors without scaling
        scales = torch.ones(s_x.size(), dtype=s_x.dtype).to(s_x.device)
        assert scales.shape == s_x.shape
        scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

        for fc in linears2scale:
            scales = scales.to(fc.weight.device)
            # Scale up the values of the weight channels
            fc.weight.mul_(scales)
            fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size)
            # Step 3: Scale back down the values of the weight channels
            fc.weight.data = fc.weight.data / scales

        out = block(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]

        original_loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
        
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            # Calculate the scales by the formula: scales = s_x^ratio
            scales = torch.pow(s_x, ratio).clamp(min=1e-4)
            assert scales.shape == s_x.shape
            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

            for fc in linears2scale:
                scales = scales.to(fc.weight.device)
                # Scale up the values of the weight channels
                fc.weight.mul_(scales)
                fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size)
                # Step 3: Scale back down the values of the weight channels
                fc.weight.data = fc.weight.data / scales

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            print(history)
            raise Exception

        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach(), (1 - best_error / original_loss)

    # onyl do AWQ for qkv projection in transformer layers
    inp = input_feat[name + '.self_attn.q_proj']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
    qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    final_scales, loss_reduction = _search_module_scale(module, qkv, inp)
    scale_ln_fcs(module.self_attn_layer_norm, qkv, final_scales)
    
    return final_scales.cpu(), loss_reduction

def model_weight_auto_scale(model, input_feat, w_bit=4, q_group_size=-1):
    from tinyfold.models.transformer import TransformerLayer

    best_scale_dict = dict()
    all_loss_reduction = []
    for name, module in model.named_modules():
        if isinstance(module, TransformerLayer) and ".0" not in name:
            best_scale_dict[name], loss_reduction = auto_scale_block(module, name, input_feat, w_bit, q_group_size)
            all_loss_reduction.append(loss_reduction)
    print(f"Average quantization error reduction across all transformer layers with AWQ: {sum(all_loss_reduction) / len(all_loss_reduction):.0%}")        
    return best_scale_dict
    
def get_calib_feat_and_weight(model, data_path):
    dataset = pickle.load(open(data_path, 'rb'))
    torch.set_grad_enabled(False)
    input_dict = dict()
    weight_dict = dict()
    def _stat_input_max_hook(m, x, y, name):
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
            if any(kqv in name for kqv in [".q", ".k", ".v"]) and ".0" not in name:
                weight_dict[name] = m.weight.abs().cpu()
                hooks.append(
                    m.register_forward_hook(
                        partial(_stat_input_max_hook, name=name)))
            
    for data in dataset:
        seq = data['seq']
        model.infer(seq)
    for hook in hooks:
        hook.remove()
    
    return input_dict, weight_dict   

def visualize(name, status, weights, activations, output_dir):
    qkv = ["q", "k", "v"]
    for i, w in enumerate(weights):
        w = w.numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x = np.arange(w.shape[1])
        y = np.arange(w.shape[0])
        x, y = np.meshgrid(x, y)
        surf = ax.plot_surface(x, y, w, cmap=cm.coolwarm, vmin=0, vmax = 0.5, linewidth=0, antialiased=False)
        ax.set_xlabel("Out Channel")
        ax.set_ylabel("In Channel")
        ax.set_zlim(0, 6)
        # fig.colorbar(surf, shrink=0.4, aspect=5)
        plt.savefig(output_dir / f"{name}.self_attn.{qkv[i]}_proj.weight.{status}_AWQ.png")
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(activations.shape[1])
    y = np.arange(activations.shape[0])
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, activations, cmap=cm.coolwarm, vmin=0, vmax = 0.125, linewidth=0, antialiased=False)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Token")
    ax.set_zlim(0, 6)
    # fig.colorbar(surf, shrink=0.4, aspect=5)
    plt.savefig(output_dir / f"{name}.self_attn.qkv_proj.activation.{status}_AWQ.png")
    
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
    visualize(name=layer, status="before", weights=weights, activations=activations, output_dir=output_dir)
    
    best_scales = model_weight_auto_scale(model, input_feat=input_dict, w_bit=w_bit)
    
    for w in weights:
        w *= best_scales[layer]
    activations /= best_scales[layer]
    visualize(name=layer, status="after", weights=weights, activations=activations, output_dir=output_dir)

if __name__ == "__main__":
    main()
