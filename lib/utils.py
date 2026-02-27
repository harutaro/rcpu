import torch
import torch.nn as nn
import os


def check_sparsity(model, threshold=1e-6):
    """
    FLAPの数え方に準拠する
    ゼロうめの場合(unstr)を想定
    """
    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    total_params = 0
    near_zero_params = 0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            near_zero_params += torch.count_nonzero(W.abs() < threshold).item()
            if 'self_attn' in name:
                total_params += hidden_size * hidden_size
            else:
                total_params += hidden_size * intermediate_size

    zero_ratio = near_zero_params / total_params
    return zero_ratio


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prepare_calibration_text_input(model, dataloader, device):
    """
    NOTE: 途中経過があれば使う, なければ保存, ただし, dataloaderの作成で時間かかることが発覚
    したので, この関数自体を呼ばない実装にしてある部分があるので注意 prop16など
    """

    modelname = model.name.replace('/', '_')
    tmpdir = 'tmp_files/'
    inps_path = os.path.join(tmpdir, f"{modelname}_inps.pt")
    outs_path = os.path.join(tmpdir, f"{modelname}_outs.pt")
    am_path = os.path.join(tmpdir, f"{modelname}_am.pt")
    pe_path = os.path.join(tmpdir, f"{modelname}_pe.pt")

    if os.path.exists(inps_path) and os.path.exists(outs_path) and os.path.exists(am_path) and os.path.exists(pe_path):
        print('loading shortcut data....')
        inps = torch.load(inps_path)
        outs = torch.load(outs_path)
        attention_mask = torch.load(am_path)
        position_embeddings = torch.load(pe_path)
        return inps, outs, attention_mask, position_embeddings

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    N = len(dataloader)
    inps = torch.zeros((N, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            # added for latest version
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    model.config.use_cache = use_cache

    print('saving shortcut data...')
    torch.save(inps, inps_path)
    torch.save(outs, outs_path)
    torch.save(attention_mask, am_path)
    torch.save(position_embeddings, pe_path)

    return inps, outs, attention_mask, position_embeddings #, position_ids
