#!/bin/bash

#  resnet extract feature
python3 pre_processing/extract_patches_features.py \
    --ref_file examples/ref_file.csv \
    --patch_data_path examples/Patches_hdf5 \
    --feature_path examples/features \
    --max_patch_number 4000 \
    --feat_type uni


