# !/bin/bash
set -x

torchrun \
    --nproc_per_node ${nproc_per_node} \
    --nnodes ${nnodes} \
    --node_rank ${node_rank} \
    --master_addr ${master_addr} \
    --master_port ${master_port} \
    autoregressive/train/extract_codes_c2i.py \
    --vq-ckpt "" \
    --data-path "" \
    --code-path "" \
    --ten-crop \
    --crop-range 1.1 \
    --image-size 384 \
