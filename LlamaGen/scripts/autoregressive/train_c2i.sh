# !/bin/bash
set -x

torchrun \
    --nproc_per_node ${nproc_per_node} \
    --nnodes ${nnodes} \
    --node_rank ${node_rank} \
    --master_addr ${master_addr} \
    --master_port ${master_port} \
    autoregressive/train/train_c2i.py \
    --cloud-save-path "" \
    --results-dir "" \
    --code-path "" \
    --image-size 384 \
    --epochs 100 \
    --ckpt-every 25000 \
    --gpt-model GPT-XL \
    --quant-method complex_phase_v2_reorder_2 \
    --skip-output-layer \
    --qat-from-base-model "" \
    --lr 1e-5 \
    --no-cloud-save \
    --global-batch-size 256 \
    # --ema \
