#!/bin/bash
#############################################################################

python -u plot.py \
    --model '/srv/share/akrishna/bias-discovery/experiments/coco/coco_densenet201.pth.tar' \
    --model_type 'densenet' \
    --model_specific_type 'densenet201' \
    --gradcam_layer 'module_features_denseblock4_denselayer32' \
    --biased_category 'skateboard' \
    --tree_json 'clustering_output/fc_skateboard_pred/treeData.json' \
    --cluster_id 3 \
    --output_path 'clustering_output/fc_skateboard_pred/cluster3.png' 

#############################################################################