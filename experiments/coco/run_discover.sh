#!/bin/bash
#############################################################################

# declare -a arr=(
#                 "person" "bicycle" "car" "motorcycle" "airplane" "bus" "train" "truck" "boat" "traffic light" 
#                 "fire hydrant" "stop sign" "parking meter" "bench" "bird" "cat" "dog" "horse" "sheep" "cow" "elephant"
#                 "bear" "zebra" "giraffe" "backpack" "umbrella" "handbag" "tie" "suitcase" "frisbee" "skis" "snowboard"
#                 "sports ball" "kite" "baseball bat" "baseball glove" "skateboard" "surfboard" "tennis racket" "bottle"
#                 "wine glass" "cup" "fork" "knife" "spoon" "bowl" "banana" "apple" "sandwich" "orange" "broccoli" "carrot" 
#                 "hot dog" "pizza" "donut" "cake" "chair" "couch" "potted plant" "bed" "dining table" "toilet" "tv"
#                 "laptop" "mouse" "remote" "keyboard" "cell phone" "microwave" "oven" "toaster" "sink" "refrigerator" 
#                 "book" "clock" "vase" "scissors" "teddy bear" "hair drier" "toothbrush"
#             )

declare -a arr=(
                "bed" "person" "cup" "airplane"
            )
for i in "${arr[@]}"
    do 
        echo $i
        python -u discover_multi.py \
        --model "/coc/pskynet1/akrishna/bias-discovery/experiments/coco/coco_densenet201.pth.tar" \
        --model_type "densenet" \
        --model_specific_type "densenet201" \
        --gradcam_layer "module_features_denseblock4_denselayer32" \
        --biased_category "$i" \
        --clustering_output_dir "clustering_output/coco/"
    done

#############################################################################