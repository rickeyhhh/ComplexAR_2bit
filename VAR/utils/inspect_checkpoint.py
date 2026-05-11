#!/usr/bin/env python3
"""
Inspect a PyTorch checkpoint and print each state_dict entry with shape,
statistics and a short guess of its purpose (reason).

Usage:
    python utils/inspect_checkpoint.py /path/to/your/checkpoint.pth
"""
import argparse
import torch
import math


def reason_for_key(key: str) -> str:
    k = key.lower()
    if 'weight_quantizer' in k or 'quant' in k or ('scale' in k and 'zero' in k):
        return 'Quantizer parameters (scale / zero_point)'
    if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
        return 'BatchNorm / running statistics'
    if k.endswith('.weight'):
        if '.conv' in k or 'conv' in k:
            return 'Convolutional layer weight (filters/kernels)'
        if '.embed' in k or 'embedding' in k:
            return 'Embedding matrix'
        return 'Linear / affine layer weight matrix'
    if k.endswith('.bias'):
        return 'Bias term'
    if 'norm' in k and ('weight' in k or 'bias' in k):
        return 'LayerNorm / GroupNorm learnable parameters'
    if 'mask' in k or 'buffer' in k:
        return 'Mask or buffer (non-trainable state)'
    if 'state_dict' in k:
        return 'Sub-module / nested state_dict'
    return 'Other / unrecognized (may be optimizer state, hyperparams, or metadata)'


def print_tensor_info(name, tensor):
    info = []
    try:
        numel = tensor.numel()
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        info.append(f'shape={shape}')
        info.append(f'dtype={dtype}')
        info.append(f'numel={numel}')

        # compute lightweight stats on CPU
        t = tensor.detach().to(torch.float32).cpu()
        if numel == 0:
            info.append('empty')
        else:
            # For extremely large tensors avoid expensive ops
            if numel > 50_000_000:
                info.append('too_large_for_stats')
            else:
                mean = float(t.mean())
                std = float(t.std(unbiased=False))
                mn = float(t.min())
                mx = float(t.max())
                info.append(f'mean={mean:.6g}')
                info.append(f'std={std:.6g}')
                info.append(f'min={mn:.6g}')
                info.append(f'max={mx:.6g}')
                # if discrete-looking values (small integer set), show unique counts up to small limit
                if numel <= 1_000_000 and t.numel() <= 1_000_000:
                    # check if values are near small ints
                    rounded = t.round()
                    if torch.allclose(t, rounded, atol=1e-6):
                        uniques = torch.unique(rounded)
                        if uniques.numel() <= 21:
                            info.append(f'unique_values={uniques.tolist()}')
    except Exception as e:
        info.append(f'ERROR_COMPUTING_STATS:{e}')
    print(f"- {name}: {' | '.join(info)}")
    print(f"  Reason: {reason_for_key(name)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', help='path to checkpoint (.pth/.pt)')
    parser.add_argument('--show-n', type=int, default=0,
                        help='if >0, show the first N tensor values for small tensors')
    args = parser.parse_args()

    data = torch.load(args.ckpt, map_location='cpu')
    # checkpoint might be a raw state_dict or a dict containing state_dict
    if isinstance(data, dict) and ('state_dict' in data or 'model_state_dict' in data):
        key = 'state_dict' if 'state_dict' in data else 'model_state_dict'
        state = data[key]
        print(f'Loaded checkpoint dict, found key "{key}" with {len(state)} entries')
    elif isinstance(data, dict) and all(isinstance(v, torch.Tensor) or isinstance(v, (int, float, list, tuple)) for v in data.values()):
        state = data
        print(f'Loaded checkpoint state_dict-like with {len(state)} entries')
    else:
        # unknown format: print top-level keys
        print('Checkpoint top-level keys:')
        for k in sorted(data.keys() if isinstance(data, dict) else []):
            print(' -', k, type(data[k]))
        # try to find nested state_dict heuristically
        state = None
        for k, v in (data.items() if isinstance(data, dict) else []):
            if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
                print(f'Found nested state_dict at key: {k} (len={len(v)})')
                state = v
                break
        if state is None:
            print('No state_dict-like structure found. Exiting.')
            return

    # Print summary
    keys = sorted(state.keys())
    print(f'Total state entries: {len(keys)}')

    for name in keys:
        val = state[name]
        if isinstance(val, torch.Tensor):
            print_tensor_info(name, val)
            if args.show_n > 0 and val.numel() <= args.show_n:
                print('  sample_values =', val.flatten().tolist())
        else:
            print(f"- {name}: type={type(val)} value={val}")
            print(f"  Reason: {reason_for_key(name)}")


if __name__ == '__main__':
    main()
