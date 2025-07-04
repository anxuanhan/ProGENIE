#!/usr/bin/bash

python3 pre_processing/create_patches.py \
        --ref_file ./examples/ref_file.csv  \  # Path to the reference file
        --wsi_path ./wsi \  # Path to the directory containing whole slide images
        --patch_path ./examples/Patches_hdf5 \  # Path to save the extracted patches
        --mask_path ./examples/Patches_hdf5 \  # Path to save the masks 
        --patch_size 256 \  # Size of the patches to be extracted
        --max_patches_per_slide 4000 # Maximum number of patches to extract per slide


