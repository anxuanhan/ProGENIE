#!/bin/bash



# Run the dataset preparation script
python prepare_dataset.py \
    --base_path examples/features_uni \
    --label_path examples/true_label \
    --save_path examples/dataset \
    --output_name independent_dataset 

  
