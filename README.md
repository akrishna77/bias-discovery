# Unsupervised Discovery of Bias in Deep Visual Recognition Models

Arvind Krishnakumar, Viraj Prabhu, Sruthi Sudhakar, Judy Hoffman

Deep learning models have been shown to learn spurious correlations from data that sometimes lead to systematic failures for certain subpopulations. Prior work has typically diagnosed this by crowdsourcing annotations for various protected attributes and measuring performance, which is both expensive to acquire and difficult to scale. In this work, we propose UDIS, an unsupervised algorithm for surfacing and analyzing such failure modes. UDIS identifies subpopulations via hierarchical clustering of dataset embeddings and surfaces systematic failure modes by visualizing low performing clusters along with their gradient-weighted class-activation maps. We show the effectiveness of UDIS in identifying failure modes in models trained for image classification on the CelebA and MSCOCO datasets. 

This repo contains the original PyTorch implementation of UDIS introduced in the following [paper](https://arxiv.org/abs/2110.15499).

## Table of Contents
1. [Setup](#setup)
2. [Discover](#discover)
3. [Visualization of cluster tree](#vis)
4. [UDIS](#udis)
5. [Using UDIS](#using)

## Setup <a name="setup"></a>

Create conda environment.
```
conda env create -f udis_env.yml
conda activate udis_env
pip install -e .
```

Download the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.
```
mkdir celeba && cd celeba
wget https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ
wget https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view?usp=sharing&resourcekey=0-YW2qIuRcWHy_1C2VaRGL3Q
```

Download the [COCO](https://cocodataset.org/#download) dataset and update `COCO_ROOT` in `experiments/coco/cluster_utils.py`.

## Discover <a name="discover"></a>

Run the following command after changing parameters as required:
```
./run_discover.sh
```

The following torchvision models are supported: alexnet, vgg, resnet, densenet, squeezenet.

## Visualization of cluster tree <a name="vis"></a>

Copy the cluster tree structure, i.e. `fc_treeData.json` into the `visualization` directory and run:

```
python -m http.server
```

## UDIS <a name="udis"></a>

Run the following commands to deploy the tool:
```
cd udis/
export FLASK_ENV=development
python run.py
```

## Using UDIS <a name="using"></a>

Use `Upload JSON` to upload a cluster tree file to the tool for parsing.  
Use `Upload Model` to upload a model file to the tool for auditing.  
Use `View Clusters` to audit a model using UDIS.  

To view results from the Original CelebA setting (Experiment 1), use the following values:
CelebA, original_celeba_model_93.pth, 0.66, 93, original_fc_93.json

To view results from the Biased Black Hair CelebA setting (Experiment 2), use the following values:
CelebA, biased_black_hair_model_92.pth, 0.66, 92, black_hair_fc_92.json

To view results from the COCO setting (Experiment 3), use the following values:
COCO, coco.pth, 0.66, 58, bed_fc_58.json

## Citation
Please consider citing this paper if you find this project useful in your research.
```
@inproceedings{2021BMVC_UDIS, 
  author = {Krishnakumar, Arvindkumar and Prabhu, Viraj and Sudhakar, Sruthi and Hoffman, Judy},
  title = {UDIS: Unsupervised Discovery of Bias in Deep Visual Recognition Models},
  year = 2021,
  booktitle = {British Machine Vision Conference (BMVC)}
}
```
