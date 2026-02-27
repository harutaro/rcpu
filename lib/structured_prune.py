import torch
import torch.nn as nn
from tqdm import tqdm
from lib.utils import find_layers, prepare_calibration_text_input
from lib.data import get_loaders
import math
import os
import math


class ActivationMeanHook:
    """
    モジュールの入力tensorの平均をrunning_meanの形で計算するクラス
    """
    def __init__(self, save_tensor=False):
        self.count = 0
        self.running_mean = None
        self.save_tensor = save_tensor
        self.x_list = []
        self.y_list = []

    def __call__(self, module, inp, out):
        size_per_batch = inp[0].shape[0]
        batch_mean = inp[0].squeeze(0).abs().mean(dim=0)

        if self.running_mean is None:
            self.running_mean = batch_mean
            self.count = size_per_batch
        else:
            total = self.count + size_per_batch
            self.running_mean = (self.running_mean * self.count + batch_mean * size_per_batch) / total
            self.count = total

        if self.save_tensor:
            self.x_list.append(inp[0].detach().cpu())  # inpはタプル
            self.y_list.append(out.detach().cpu())  # outはテンソル

    def get_mean_activation(self):
        return self.running_mean if self.running_mean is not None else None
    
    def get_x_all(self):
        return torch.cat(self.x_list, dim=0)

    def get_y_all(self):
        return torch.cat(self.y_list, dim=0)
    
    def reset(self):
        self.count = 0
        self.running_mean = None
        self.x_list = []
        self.y_list = []


def get_layers(model):
    if hasattr(model, "language_model"):  # MLLM
        return model.language_model.model.layers
    else:
        return model.model.layers


def compress(layer, attn_mask, mlp_mask, unstr=True):
    if unstr:  # for evaluation
        if attn_mask is not None:
            layer.self_attn.o_proj.weight.data *= attn_mask
            layer.self_attn.q_proj.weight.data *= attn_mask.unsqueeze(-1)
            layer.self_attn.k_proj.weight.data *= attn_mask.unsqueeze(-1)
            layer.self_attn.v_proj.weight.data *= attn_mask.unsqueeze(-1)
        if mlp_mask is not None:
            layer.mlp.down_proj.weight.data *= mlp_mask
            layer.mlp.up_proj.weight.data *= mlp_mask.unsqueeze(-1)
            layer.mlp.gate_proj.weight.data *= mlp_mask.unsqueeze(-1)
    else: # real pruning. 
        if attn_mask is not None:
            change = torch.ones_like(attn_mask)
            change[1:] = attn_mask[1:] != attn_mask[:-1]
            head_mask = attn_mask[change]
            retain_heads = head_mask.sum()
            head_dim = attn_mask.shape[0] / head_mask.shape[0]

            layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.q_proj.out_features = attn_mask.sum().item()
            layer.self_attn.k_proj.out_features = attn_mask.sum().item()
            layer.self_attn.v_proj.out_features = attn_mask.sum().item()
            output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]
            layer.self_attn.o_proj.weight.data = output_weight

            layer.self_attn.num_heads = retain_heads
            layer.self_attn.hidden_size = retain_heads * head_dim

        if mlp_mask is not None:
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
            layer.mlp.up_proj.out_features = mlp_mask.sum().item()
            layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
            output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]  
            layer.mlp.down_proj.weight.data = output_weight
            
            layer.mlp.intermediate_size = mlp_mask.sum().item()


def compute_mask(W_metric, name, params, return_score=False):
    # based on flap
    num_heads = params['num_heads']
    head_dim = params['head_dim']
    pruning_ratio = params['pruning_ratio']
    if name == 'self_attn.o_proj':
        W_metric_head = W_metric.reshape(-1, head_dim).sum(dim=1) # importance score of each head
        thresh = torch.sort(W_metric_head.cuda())[0][int(pruning_ratio*num_heads)].cpu()
        head_mask = (W_metric_head>=thresh)
        W_mask = head_mask.repeat_interleave(head_dim)

        if return_score:
            score_head = W_metric_head * head_mask
            score = score_head.repeat_interleave(head_dim)
            return W_mask, score
        else:
            return W_mask

    else:
        thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*pruning_ratio)].cpu()
        W_mask = (W_metric>=thresh)
        if return_score:
            score = W_metric * W_mask
            return W_mask, score
        else:
            return W_mask


def prune_magnitude(args, model, processor, device):
    layers = get_layers(model)
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})
        params = {
            'num_heads': model.config.num_attention_heads, 
            'head_dim': model.config.head_dim,
            'pruning_ratio': args.pruning_ratio
            }
        for name in subset:
            W_metric = torch.norm(subset[name].weight.data, dim=0)
            W_mask = compute_mask(W_metric, name, params)
            if name == 'self_attn.o_proj':
                compress(layer, W_mask, None, unstr=args.unstr)
            else:
                compress(layer, None, W_mask, unstr=args.unstr)


def prune_wanda(args, model, processor, device):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed,
                                  seqlen=model.seqlen, tokenizer=processor)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_embeddings = prepare_calibration_text_input(model, dataloader, device)

    layers = get_layers(model)
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        # 入力を捕捉するhookの準備
        hook_map = {}
        handles = []
        for name in subset:  
            hook = ActivationMeanHook(save_tensor=True)
            handles.append(subset[name].register_forward_hook(hook))
            hook_map[name] = hook

        # キャリブレーションデータを流してフックを起動し、何もしなかった場合の層の出力を取得
        for j in range(len(dataloader)):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
        # hookを適用解除（クラスは消えない）
        for h in handles:
            h.remove()
        
        # wanda pruning
        for name in subset:
            x = hook_map[name].get_mean_activation()
            W_metric = (torch.abs(subset[name].weight.data) * x).sum(dim=0)
            params = {
                'num_heads': model.config.num_attention_heads, 
                'head_dim': model.config.head_dim,
                'pruning_ratio': args.pruning_ratio
                }
            W_mask = compute_mask(W_metric, name, params)

            if name == 'self_attn.o_proj':
                compress(layer, W_mask, None, unstr=args.unstr)
            else:
                compress(layer, None, W_mask, unstr=args.unstr)

        # pruning後の出力を計算
        for j in range(len(dataloader)):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
        inps, outs = outs, inps


def prune_rcpu(args, model, processor, device):
    model.data_dict['k_rcpu'] = 0
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed,
                                  seqlen=model.seqlen, tokenizer=processor)
    with torch.no_grad():
        inps, outs, attention_mask, position_embeddings = prepare_calibration_text_input(model, dataloader, device)
    print("dataset loading complete")

    print('pruning and update all started...')
    layers = get_layers(model)
    import time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        # 入力を捕捉するhookの準備
        hook_map = {}
        handles = []
        for name in subset:  
            hook = ActivationMeanHook(save_tensor=True)
            handles.append(subset[name].register_forward_hook(hook))
            hook_map[name] = hook

        # キャリブレーションデータを流してフックを起動し、何もしなかった場合の層の出力を取得
        for j in range(len(dataloader)):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
        # hookを適用解除（クラスは消えない）
        for h in handles:
            h.remove()
        
        Y_dense = {name: hook_map[name].get_y_all() for name in subset}
        # pruning
        for name in subset:
            params = {
                'num_heads': model.config.num_attention_heads, 
                'head_dim': model.config.head_dim,
                'pruning_ratio': args.pruning_ratio # rho_oにしてもおｋ
                }
            W = subset[name].weight.data
            X = hook_map[name].get_x_all()
            X = X.view(-1, X.shape[-1]).transpose(0, 1).to(device)  # [d_in, N]
            Y = Y_dense[name]
            Y = Y.view(-1, Y.shape[-1]).transpose(0, 1).to(device)  # [d_out, N]

            # --- γ_j = ||w_j|| * E[|x_j|] * Var(x_j) ---
            mu_x = X.mean(dim=1, keepdim=True)                          # [d_in, 1]
            ex2  = (X * X).mean(dim=1)                                   # [d_in]
            var_x = torch.clamp(ex2 - (mu_x.squeeze(1) ** 2), min=0.0)   # [d_in]
            col_l2 = torch.linalg.norm(W, dim=0)                         # ||w_j||, [d_in]
            mean_abs_x = X.abs().mean(dim=1)                             # E[|x_j|], [d_in]
            gamma = col_l2 * mean_abs_x * var_x                          # [d_in]


            W_mask = compute_mask(gamma, name, params)

            keep  = W_mask.nonzero(as_tuple=True)[0]
            drop  = (~W_mask).nonzero(as_tuple=True)[0]

            # -------- Procrustes -----------------
            W_p = W[:, keep]                         # [d_out, d']
            X_p = X[keep, :]                         # [d', N]

            A   = W_p @ X_p                          # [d_out, N]
            A = A.float()  # 32bit
            Y = Y.float()
            U, S, Vh = torch.linalg.svd(Y @ A.T, full_matrices=False)

            Q = U @ Vh
            Q = Q.to(model.dtype)
            QWp = Q @ W_p
            W_rot = QWp

            subset[name].weight.data[:, W_mask] = W_rot

            if name == 'self_attn.o_proj':
                compress(layer, W_mask, None, unstr=args.unstr)
            else:
                compress(layer, None, W_mask, unstr=args.unstr)

        # pruning後の出力を計算
        for j in range(len(dataloader)):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
        inps, outs = outs, inps
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"loop took {elapsed:.6f} s")
    print(f'elapsed / 32 = {elapsed / 32}')


def prune_rcpu_scaled(args, model, processor, device):
    model.data_dict['k_rcpu'] = 0
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed,
                                  seqlen=model.seqlen, tokenizer=processor)
    with torch.no_grad():
        inps, outs, attention_mask, position_embeddings = prepare_calibration_text_input(model, dataloader, device)
    print("dataset loading complete")

    print('pruning and update all started...')
    layers = get_layers(model)
    import time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        # 入力を捕捉するhookの準備
        hook_map = {}
        handles = []
        for name in subset:  
            hook = ActivationMeanHook(save_tensor=True)
            handles.append(subset[name].register_forward_hook(hook))
            hook_map[name] = hook

        # キャリブレーションデータを流してフックを起動し、何もしなかった場合の層の出力を取得
        for j in range(len(dataloader)):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
        # hookを適用解除（クラスは消えない）
        for h in handles:
            h.remove()
        
        Y_dense = {name: hook_map[name].get_y_all() for name in subset}
        # pruning
        for name in subset:
            params = {
                'num_heads': model.config.num_attention_heads, 
                'head_dim': model.config.head_dim,
                'pruning_ratio': args.pruning_ratio # rho_oにしてもおｋ
                }
            W = subset[name].weight.data
            X = hook_map[name].get_x_all()
            X = X.view(-1, X.shape[-1]).transpose(0, 1).to(device)  # [d_in, N]
            Y = Y_dense[name]
            Y = Y.view(-1, Y.shape[-1]).transpose(0, 1).to(device)  # [d_out, N]

            # --- γ_j = ||w_j|| * E[|x_j|] * Var(x_j) ---
            mu_x = X.mean(dim=1, keepdim=True)                          # [d_in, 1]
            ex2  = (X * X).mean(dim=1)                                   # [d_in]
            var_x = torch.clamp(ex2 - (mu_x.squeeze(1) ** 2), min=0.0)   # [d_in]
            col_l2 = torch.linalg.norm(W, dim=0)                         # ||w_j||, [d_in]
            mean_abs_x = X.abs().mean(dim=1)                             # E[|x_j|], [d_in]
            gamma = col_l2 * mean_abs_x * var_x                          # [d_in]


            W_mask = compute_mask(gamma, name, params)

            keep  = W_mask.nonzero(as_tuple=True)[0]
            drop  = (~W_mask).nonzero(as_tuple=True)[0]

            # -------- Procrustes -----------------
            W_p = W[:, keep]                         # [d_out, d']
            X_p = X[keep, :]                         # [d', N]

            A   = W_p @ X_p                          # [d_out, N]
            A = A.float()  # 32bit
            Y = Y.float()
            U, S, Vh = torch.linalg.svd(Y @ A.T, full_matrices=False)
            Q = U @ Vh
            Q = Q.to(model.dtype)
            QWp = Q @ W_p
            
            s_uni = S.sum() / (A.norm()**2 + 1e-6)  # scaling
            W_rot = s_uni * QWp

            subset[name].weight.data[:, W_mask] = W_rot

            if name == 'self_attn.o_proj':
                compress(layer, W_mask, None, unstr=args.unstr)
            else:
                compress(layer, None, W_mask, unstr=args.unstr)

        # pruning後の出力を計算
        for j in range(len(dataloader)):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
        inps, outs = outs, inps
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"loop took {elapsed:.6f} s")
    print(f'elapsed / 32 = {elapsed / 32}')


def prune_prop_ls(args, model, processor, device):
    model.config.use_cache = False 
    model.data_dict['lam'] = args.lam
    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed,
                                  seqlen=model.seqlen, tokenizer=processor)
    with torch.no_grad():
        inps, outs, attention_mask, position_embeddings = prepare_calibration_text_input(model, dataloader, device)
    print("dataset loading complete")

    print('pruning and update all started...')
    layers = get_layers(model)
    df_ls = 0
    df_rcpu = 0

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        # 入力を捕捉するhookの準備
        hook_map = {}
        handles = []
        for name in subset:  
            hook = ActivationMeanHook(save_tensor=True)
            handles.append(subset[name].register_forward_hook(hook))
            hook_map[name] = hook

        # キャリブレーションデータを流してフックを起動し、何もしなかった場合の層の出力を取得
        for j in range(len(dataloader)):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
        # hookを適用解除（クラスは消えない）
        for h in handles:
            h.remove()
        
        Y_dense = {name: hook_map[name].get_y_all() for name in subset}
        # pruning
        for name in subset:
            params = {
                'num_heads': model.config.num_attention_heads, 
                'head_dim': model.config.head_dim,
                'pruning_ratio': args.pruning_ratio # rho_oにしてもおｋ
                }
            W = subset[name].weight.data
            X = hook_map[name].get_x_all()
            X = X.view(-1, X.shape[-1]).transpose(0, 1).to(device)  # [d_in, N]
            Y = Y_dense[name]
            Y = Y.view(-1, Y.shape[-1]).transpose(0, 1).to(device)  # [d_out, N]

            mu_x = X.mean(dim=1, keepdim=True)                          # [d_in, 1]
            ex2  = (X * X).mean(dim=1)                                   # [d_in]
            var_x = torch.clamp(ex2 - (mu_x.squeeze(1) ** 2), min=0.0)   # [d_in]
            col_l2 = torch.linalg.norm(W, dim=0)                         # ||w_j||, [d_in]
            mean_abs_x = X.abs().mean(dim=1)                             # E[|x_j|], [d_in]
            gamma = col_l2 * mean_abs_x * var_x                          # [d_in]


            W_mask = compute_mask(gamma, name, params)

            W_keep = W[:, W_mask]
            X_dash = X[W_mask, :]

            XXt = X_dash @ X_dash.T
            XXt = XXt + args.lam*torch.eye(XXt.shape[0], device=XXt.device, dtype=XXt.dtype)
            XXt = XXt.to(device=device, dtype=torch.float32)
            B = (Y @ X_dash.T + args.lam*W_keep) @ torch.linalg.inv(XXt).to(model.dtype)
            subset[name].weight.data[:, W_mask] = B

            if name == 'self_attn.o_proj':
                compress(layer, W_mask, None, unstr=args.unstr)
            else:
                compress(layer, None, W_mask, unstr=args.unstr)

        # pruning後の出力を計算
        for j in range(len(dataloader)):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
        inps, outs = outs, inps

