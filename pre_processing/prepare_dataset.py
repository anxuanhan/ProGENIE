import os
import sys
import h5py
import torch
import argparse
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare test dataset from feature and label files.")
    parser.add_argument('--base_path', type=str, required=True,
                        help="Path to feature .h5 files, each sample in a separate directory.")
    parser.add_argument('--label_path', type=str, required=True,
                        help="Path to CSV file containing expression labels.")
    parser.add_argument('--save_path', type=str, required=True,
                        help="Path to save the output .pt dataset.")
    parser.add_argument('--output_name', type=str, default='test_dataset.pt',
                        help="Name of the output file (default: test_dataset.pt)")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.label_path):
        print(f"Error: Label file does not exist: {args.label_path}")
        sys.exit(1)

    if not os.path.exists(args.base_path):
        print(f"Error: Feature base path does not exist: {args.base_path}")
        sys.exit(1)

    os.makedirs(args.save_path, exist_ok=True)

    try:
        label_df = pd.read_csv(args.label_path, index_col=0)
        print(f"Loaded label data, shape: {label_df.shape}")
    except Exception as e:
        print(f"Failed to load label file: {e}")
        sys.exit(1)

    gene_names = list(label_df.columns)
    print(f"Number of genes: {len(gene_names)}")

    features_list, labels_list, sample_ids, failed = [], [], [], []
    processed = 0

    print("Processing samples...")
    for sample_id in tqdm(label_df.index):
        h5_path = os.path.join(args.base_path, sample_id, f"{sample_id}.h5")
        if not os.path.exists(h5_path):
            failed.append(sample_id)
            continue

        try:
            with h5py.File(h5_path, 'r') as f:
                if 'cluster_features' not in f:
                    failed.append(sample_id)
                    continue

                feat = f['cluster_features'][:]
                label = label_df.loc[sample_id].values

                features_list.append(torch.tensor(feat, dtype=torch.float32))
                labels_list.append(torch.tensor(label, dtype=torch.float32))
                sample_ids.append(sample_id)
                processed += 1

                if processed == 1:
                    print(f"Example feature shape: {feat.shape}")
                    print(f"Example label shape: {label.shape}")

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            failed.append(sample_id)

    print(f"Processed samples: {processed}")
    print(f"Failed samples: {len(failed)}")

    if not features_list:
        print("Error: No valid features loaded.")
        sys.exit(1)

    features_tensor = torch.stack(features_list)
    labels_tensor = torch.stack(labels_list)

    dataset = {
        'test': {
            'features': features_tensor,
            'labels': labels_tensor,
            'sample_ids': sample_ids
        },
        'gene_names': gene_names,
    }

    output_file = os.path.join(args.save_path, args.output_name)
    torch.save(dataset, output_file)

    print(f"Saved dataset to: {output_file}")
    print(f"Features shape: {features_tensor.shape}, Labels shape: {labels_tensor.shape}")
    if failed:
        print(f"Failed sample count: {len(failed)}")
        print(f"Example failed IDs: {failed[:5]}")

if __name__ == '__main__':
    main()
