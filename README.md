# 🧚🏻‍♀️ProGENIE: Predicting Gene Expression from Whole Slide Images in Prostate Cancer Using Deep Learning

### Abstract

*Prostate cancer exhibits complex heterogeneity, requiring costly sequencing to characterize its diversity for precision medicine. While deep learning on whole slide images (WSIs) has shown promise in clinically relevant questions, transcriptomic prediction focused on prostate cancer remains limited. Here, we develop ProGENIE, a multi-head attention-pooling framework directly predicting gene expression in prostate cancer from WSIs. Trained on The Cancer Genome Atlas (TCGA) dataset, ProGENIE demonstrates strong generalizability on an independent cohort from South Australian Hospitals (SAH), achieving a median Pearson correlation coefficient (PCC) close to 0.6 for the top 1,000 genes and accurately predicting 3,167 genes with PCC > 0.4. ProGENIE accurately predicts gene expression associated with prostate cancer development and reliably characterizes the tumor microenvironment. Furthermore, the predicted transcriptomic profiles strongly correlate with drug sensitivity and immunotherapy response. This cost-effective approach links tissue morphology to molecular profiles and supports personalized treatment in prostate cancer.*

### Overview
<img src="https://github.com/anxuanhan/ProGENIE/blob/main/model_architecture.png" alt="ChatGPT" width="600"/>

## Folder structure

## Pre-requisites

## Installation Guide for Linux (using anaconda)
1. Clone this git repository: `git clone https://github.com/anxuanhan/ProGENIE.git`
2. `cd ProGENIE`
3. Create a conda environment: `conda create -n progenie python=3.9`
4. `conda activate progenie
5. Install the required package dependencies: `pip install -r requirements.txt`


## 准备工作
1. 准备wsi文件夹，里面储存着WSI文件
2. 在example文件夹下，准备ref_file.csv文件，eg.

| WSI File Name      | Patient ID       | rna_KRT5 | rna_ACTB | ...       | rna_SCGB2A1  | tcga_project |
|-------------------|------------------|----------|----------|-----------|------------|---------------|
| TCGA-2A-A8VL-01A  | TCGA-2A-A8VL-01A | 5.534787 | 9.443552 | ...       | 2.650512  | TCGA-PRAD     |
| TCGA-2A-A8VO-01A  | TCGA-2A-A8VO-01A | 0.213876 | 9.519404 | ...       | 0.9438838 | TCGA-PRAD     |
| TCGA-2A-A8VT-01A  | TCGA-2A-A8VT-01A | 4.868341 | 8.809999 | ...       | 1.121745 | TCGA-PRAD     |

## Preprocessing
1.创建patch
2.提取patch特征
3.得到k-Means features

## download pre-trained model on an independent datase





