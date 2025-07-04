#!/bin/bash


# uni feature clustering
python3 pre_processing/kmean_features.py \
  --ref_file examples/ref_file.csv \
  --patch_data_path examples/Patches_hdf5 \
  --feature_path examples/features_uni \
  --num_clusters 100


  
