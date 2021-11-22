# #!/bin/bash
# #############################################################################

#  python -u discover.py \
#      --dataroot '/coc/pskynet1/akrishna/bias-discovery/celeba/img_align_celeba/' \
#      --csv_file '/coc/pskynet1/akrishna/bias-discovery/celeba/list_attr_celeba.txt' \
#      --model '/coc/pskynet1/akrishna/bias-discovery/experiments/celeba/clustering_output/original_celeba/original_celeba_model_93.pth' \
#      --gradcam_layer 'layer4' \
#      --batch_size 64 \
#      --model_type 'resnet' \
#      --clustering_output_dir '/coc/pskynet1/akrishna/bias-discovery/experiments/celeba/clustering_output/original_celeba/93/' \

###########################################################################

#!/bin/bash
############################################################################

python -u discover.py \
   --dataroot '/coc/pskynet1/akrishna/bias-discovery/celeba/img_align_celeba/' \
   --csv_file '/coc/pskynet1/akrishna/bias-discovery/celeba/list_attr_celeba.txt' \
   --model '/coc/pskynet1/akrishna/bias-discovery/experiments/celeba/clustering_output/biased_celeba_black_hair/biased_black_hair_model_92.pth' \
   --gradcam_layer 'layer4' \
   --batch_size 64 \
   --model_type 'resnet' \
   --clustering_output_dir '/coc/pskynet1/akrishna/bias-discovery/experiments/celeba/clustering_output/biased_celeba_black_hair/92/' \
    
# #############################################################################
