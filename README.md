# 🧚🏻‍♀️ProGENIE: Predicting Gene Expression from Whole Slide Images in Prostate Cancer Using Deep Learning

### Abstract

*Prostate cancer exhibits complex heterogeneity, requiring costly sequencing to characterize its diversity for precision medicine. While deep learning on whole slide images (WSIs) has shown promise in clinically relevant questions, transcriptomic prediction focused on prostate cancer remains limited. Here, we develop ProGENIE, a multi-head attention-pooling framework directly predicting gene expression in prostate cancer from WSIs. Trained on The Cancer Genome Atlas (TCGA) dataset, ProGENIE demonstrates strong generalizability on an independent cohort from South Australian Hospitals (SAH), achieving a median Pearson correlation coefficient (PCC) close to 0.6 for the top 1,000 genes and accurately predicting 3,167 genes with PCC > 0.4. ProGENIE accurately predicts gene expression associated with prostate cancer development and reliably characterizes the tumor microenvironment. Furthermore, the predicted transcriptomic profiles strongly correlate with drug sensitivity and immunotherapy response. This cost-effective approach links tissue morphology to molecular profiles and supports personalized treatment in prostate cancer.*

### Overview
<p align="center">
  <img src="https://github.com/anxuanhan/ProGENIE/blob/main/pics/model_architecture.png" alt="model architecture" width="600"/>
</p>

## Pre-requisites
- Linux (Tested on Red Hat Enterprise Linux 8.4)
- NVIDIA GPU (Tested on NVIDIA A100 PCIe 40GB)
- Python (Python 3.10.0),
PyTorch==1.13.1+cu116,
Torchvision==0.14.1+cu116,
Torchaudio==0.13.1+cu116,
Matplotlib==3.10.1,
NumPy==1.23.5,
OpenCV-Python==4.11.0.86,
Openslide-Python==1.4.1,
Pandas==2.2.3,
Scikit-Image==0.25.2,
Scikit-Learn==1.6.1,
SciPy==1.15.2,
Seaborn==0.13.2,
Einops==0.8.1,
Transformers==4.49.0,
Timm==1.0.3,
Tensorboard==2.19.0,
TensorboardX==2.6.2.2

## Installation Guide for Linux (using anaconda)
1. Clone this git repository: `git clone https://github.com/anxuanhan/ProGENIE.git`
2. `cd ProGENIE`
3. Create a conda environment: `conda create -n progenie python=3.10.0`
4. `conda activate progenie
5. Install the required package dependencies: `pip install -r requirements.txt`


## Preparation
1. Prepare the `wsi/` directory, which contains the whole slide images
2. Prepare the reference file: `example/ref_file.csv`

  For example:

| WSI File Name      | Patient ID       | rna_A1BG | rna_A2M | ...       | rna_ZZZ3  | tcga_project |
|-------------------|------------------|----------|----------|-----------|------------|---------------|
| TCGA-2A-A8VL-01A  | TCGA-2A-A8VL-01A | 0.0658   | 5.1469 | ...       | 2.4027  | TCGA-PRAD     |
| TCGA-2A-A8VO-01A  | TCGA-2A-A8VO-01A | 0.0243   | 7.0980  | ...       | 2.5807 | TCGA-PRAD     |
| TCGA-2A-A8VT-01A  | TCGA-2A-A8VT-01A | 0.0195   | 5.5461 | ...       | 3.6254 | TCGA-PRAD     |

3. Prepare the ground truth label file: `examples/true_label.csv`
   

## Preprocessing
**1. Create patches from WSIs**
   
   To extract image patches from raw Whole Slide Images (WSIs), run the patch generation script provided in: `pre_processing/create_patches.py`
   
   An example script to run the patch extraction: `scripts/create_patches.sh`
   
**2. Extract Features from Patches**

   To extract features from patches using a pretrained encoder, run the feature extraction script provided in: `pre_processing/extract_patch_features.py`
   
   An example script to run the feature extraction:`scripts/extract_patches_features.sh`

   Note: Pretrained encoder weights (e.g., for UNI, CHIEF, and Prov-GigaPath) can be obtained by referring to the following repositories:
  
   - UNI: https://github.com/mahmoodlab/UNI

   - CHIEF: https://github.com/hms-dbmi/CHIEF

   - Prov-GigaPath: https://github.com/prov-gigapath/prov-gigapath
   
**3. Obtain k-Means Features**

  To compute k-Means features from extracted patch features, run the clustering script provided in: `pre_processing/kmean_features.py`

  An example script to run the k-Means clustering: `scripts/kmean_features.sh`

## Inferrence on independent dataset
We released the model weights for four pre-trained models on [HuggingFace](https://huggingface.co/ananananxuan/ProGENIE/tree/main "HuggingFace"), please download the weights first.

**1. Prepare the dataset**

To combine the k-Means features with ground truth gene expression profiles for model training, run the dataset preparation script provided in: `pre_processing/prepare_dataset.py`

An example script to run the dataset preparation: `scripts/prepare_dataset.sh`

**2. inference and evaluation**

To perform model inference on the test set and evaluate performance:

Run the main inference and evaluation script:`inference.py`

You can find an example script here:`scripts/inference.sh`

The output will be saved in: `examples/results`

This includes:

- test_pred_labels.csv: predicted gene expression values

- test_true_labels.csv: ground truth labels

- test_gene_metrics.csv: PCC, RMSE, and R²











