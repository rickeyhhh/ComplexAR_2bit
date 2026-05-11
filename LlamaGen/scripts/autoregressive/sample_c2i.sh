#!/bin/bash
set -x

torchrun \
    --nproc_per_node ${nproc_per_node} \
    --nnodes ${nnodes} \
    --node_rank ${node_rank} \
    --master_addr ${master_addr} \
    --master_port ${master_port} \
    autoregressive/sample/sample_c2i_ddp.py \
    --gpt-ckpt "" \
    --sample-dir "" \
    --image-size 384 \
    --cfg-scale 2.25 \
    --quant-method complex_phase_v2_reorder_2 \
    --skip-output-layer \
    --compile \
