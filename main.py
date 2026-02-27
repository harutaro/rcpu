import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.utils import check_sparsity
from lib.eval import eval_ppl
import subprocess
import time
import gc

save_model_dir = './models/'

def cuda_mem_mib(device=0, sync=True, do_gc=False):
    MiB = 1024**2
    if do_gc:
        gc.collect()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    alloc = int(torch.cuda.memory_allocated(device) / MiB)    # 実際に張り付いてるテンソル
    reserv = int(torch.cuda.memory_reserved(device)  / MiB)   # キャッシュ含む確保
    peak  = int(torch.cuda.max_memory_allocated(device) / MiB)
    return alloc, reserv, peak

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="huggyllama/llama-7b", help='model_id')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--method', type=str, default="rcpu")
    parser.add_argument('--pruning_ratio', type=float, default=0.2, help='Number of calibration samples.')
    parser.add_argument('--save_result', action="store_true")
    parser.add_argument('--outfile', type=str, default="xxxx", help='filename')
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--savemodelname', type=str, default='')
    parser.add_argument('--skip_pruning', action="store_true")
    parser.add_argument('--skip_eval', action="store_true")
    parser.add_argument('--unstr', action="store_true")
    parser.add_argument('--lam', type=float, default=1e-4, help='for LS')
    args = parser.parse_args()

    # GPU/CPU の設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # モデルとトークナイザーの読み込み
    if 'Llama-2' in args.model:
        processor = AutoTokenizer.from_pretrained(args.model, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=None,
            torch_dtype=torch.float32
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        processor = AutoTokenizer.from_pretrained(args.model)  # 変数名は合わせた

    model.data_dict = {}
    model.name = args.model  # 地味に使うので
    processor.name = args.model
    print(model)
    print(model.config)
    num_params_org = model.num_parameters()
    print(f'original num_params={num_params_org / 1e9:.2f}B')
    model.seqlen = 128
    model.num_params = num_params_org
    model.to(device)
    model.eval()


    # pruning
    if not args.skip_pruning:
        print('Pruning starts...')
        if args.method == 'mag':
            from lib.structured_prune import prune_magnitude
            prune_magnitude(args, model, processor, device)
        elif args.method == 'wanda':
            from lib.structured_prune import prune_wanda
            prune_wanda(args, model, processor, device)
        elif args.method == 'propls':  
            from lib.structured_prune import prune_prop_ls
            prune_prop_ls(args, model, processor, device)
        elif args.method == 'rcpu':
            from lib.structured_prune import prune_rcpu
            prune_rcpu(args, model, processor, device)
        elif args.method == 'rcpuscaled':
            from lib.structured_prune import prune_rcpu_scaled
            prune_rcpu_scaled(args, model, processor, device)
        else:
            raise NotImplementedError
    else:
        model.config.use_cache = False

    # Check the sparsity of the model
    print("*"*30)
    if args.unstr:
        sparsity_ratio = check_sparsity(model)
        print(f"zero ratio= {sparsity_ratio:.2f}")
    else:
        print('after pruning')
        alloc, reserv, peak = cuda_mem_mib()
        print(f"allocated={alloc} MiB, reserved={reserv} MiB, peak={peak} MiB")

    print("*"*30)

    if not args.skip_eval:
        print(f'{args.method} ppl evaluation started.')
        ppl = eval_ppl(model, processor, device)    
        model.data_dict['testppl'] = ppl
        print(f"ppl on wikitext {ppl}")
        
    if args.save_result:
        print('saving results...')
        text = (
            f'model:{args.model} method:{args.method} '
            f'pr:{args.pruning_ratio} ns:{args.nsamples}'
        )
        if hasattr(model, 'data_dict'):
            model.data_dict.pop('trainloader', None)
            for k in model.data_dict:
                text += f' {k}:{model.data_dict[k]}'
        text += '\n'
        filename = f'/mnt/data1/llm-pruning/tmp_results/{args.outfile}.txt'
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(text)

    if args.save_model:
        # huggingface style
        fn = f'{args.model}_{args.method}_{args.nsamples}_{args.pruning_ratio}'
        if args.savemodelname != '':
            fn = fn + f'_{args.savemodelname}'
        path = f'{save_model_dir}/{fn}'  # change
        if not os.path.exists(path):
            os.makedirs(path)
        if 'vicuna' in args.model:
            gc = model.generation_config
            gc.do_sample = True    # ← 保存時のみ
        model.save_pretrained(path)
        processor.save_pretrained(path)
        # pt style
        # path = f'{save_model_dir}/{fn}.pt'
        # torch.save(model.state_dict(), f'{path}')


if __name__ == "__main__":
    main()
