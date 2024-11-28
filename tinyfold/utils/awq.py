import pickle
import torch
from torch import nn
from functools import partial
from tinyfold.utils.quantize import pseudo_quantize_tensor
            
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

def auto_scale_block(module, name, input_feat, w_bit, q_group_size_candidates):

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

        for n, module in block.named_modules():
            if module in linears2scale:
                scales = scales.to(module.weight.device)
                # Scale up the values of the weight channels
                module.weight.mul_(scales)
                module.weight.data = pseudo_quantize_tensor(module.weight.data, w_bit, q_group_size_candidates)
                # Step 3: Scale back down the values of the weight channels
                module.weight.data = module.weight.data / scales
            elif isinstance(module, nn.Linear):
                module.weight.data = pseudo_quantize_tensor(module.weight.data, w_bit, q_group_size_candidates)

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

            for n, module in block.named_modules():
                if module in linears2scale:
                    scales = scales.to(module.weight.device)
                    # Scale up the values of the weight channels
                    module.weight.mul_(scales)
                    module.weight.data = pseudo_quantize_tensor(module.weight.data, w_bit, q_group_size_candidates)
                    # Step 3: Scale back down the values of the weight channels
                    module.weight.data = module.weight.data / scales
                elif isinstance(module, nn.Linear):
                    module.weight.data = pseudo_quantize_tensor(module.weight.data, w_bit, q_group_size_candidates)

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

def model_weight_auto_scale(model, input_feat, w_bit=4, q_group_size_candidates=[-1]):
    from tinyfold.models.transformer import TransformerLayer

    best_scale_dict = dict()
    all_loss_reduction = []
    for name, module in model.named_modules():
        if isinstance(module, TransformerLayer) and ".0" not in name:
            best_scale_dict[name], loss_reduction = auto_scale_block(module, name, input_feat, w_bit, q_group_size_candidates)
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