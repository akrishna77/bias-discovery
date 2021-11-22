#!/bin/sh
#############################################################################
python -u train.py \
    --epochs 5 \
    --weight_decay 0.0001 \
    --batch_size 64 \
    --save_dir /srv/share/akrishna/bias-discovery/experiments/celeba/clustering_output/original_celeba/ \
    --drop_rate 0.3 \
    --lr 0.0005 
#############################################################################
