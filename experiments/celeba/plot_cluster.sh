#!/bin/bash
#############################################################################

python -u plot.py \
    --dataroot '/srv/share/akrishna/bias-discovery/celeba/img_align_celeba/' \
    --csv_file '/srv/share/akrishna/bias-discovery/celeba/list_attr_celeba.txt' \
    --model '/srv/share/akrishna/bias-discovery/experiments/celeba/clustering_output/biased_celeba_male/stoic-night-130.pth' \
    --gradcam_layer 'layer4' \
    --batch_size 32 \
    --model_type 'resnet' \
    --tree_json '/srv/share/akrishna/bias-discovery/experiments/celeba/clustering_output/biased_celeba_male/fc_non_smiling/treeData.json' \
    --cluster_id 24073 \
    --output_path '/srv/share/akrishna/bias-discovery/experiments/celeba/clustering_output/biased_celeba_male/fc_non_smiling/cluster24073.png'

#############################################################################
