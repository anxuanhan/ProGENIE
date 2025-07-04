import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from scipy.stats import pearsonr
from evaluate import calculate_metrics, rmse, r2_score
from model import GeneExpressionModelHead

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = avg_loss
    return metrics

def test_model(model, dataloader, device):
    model.to(device)
    print("Evaluating model on test set...")
    metrics = evaluate_model(model, dataloader, device)
    print("=" * 60)
    print("Test Set Metrics:")
    print(f"  Loss (MSE): {metrics['loss']:.6f}")
    print(f"  RMSE      : {metrics['rmse']:.6f}")
    print(f"  PCC       : {metrics['pcc']:.4f}")
    print(f"  R²        : {metrics['r2']:.4f}")
    print("=" * 60)
    return metrics

def output_predictions(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for feature, label in dataloader:
            feature = feature.to(device)
            output = model(feature)
            preds.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return preds, labels

def compute_genewise_pcc(df_true: pd.DataFrame, df_pred: pd.DataFrame):
    assert df_true.shape == df_pred.shape
    assert list(df_true.columns) == list(df_pred.columns)

    gene_names = df_true.columns
    pcc_results, rmse_results, r2_results = [], [], []

    for gene in gene_names:
        true_vals = df_true[gene].values
        pred_vals = df_pred[gene].values

        if true_vals.std() == 0 or pred_vals.std() == 0:
            pcc, rmses, r2s = float('nan'), float('nan'), float('nan')
        else:
            pcc, _ = pearsonr(true_vals, pred_vals)
            rmses = rmse(true_vals, pred_vals)
            r2s = r2_score(true_vals, pred_vals)

        pcc_results.append(pcc)
        rmse_results.append(rmses)
        r2_results.append(r2s)

    gene_df = pd.DataFrame({
        "gene": gene_names,
        "pcc": pcc_results,
        "rmse": rmse_results,
        "r2": r2_results
    })

    print("📊 Gene-wise PCC Summary:")
    print(f"Mean PCC   : {gene_df['pcc'].mean():.4f}")
    print(f"Median PCC : {gene_df['pcc'].median():.4f}")
    print(f"Max PCC    : {gene_df['pcc'].max():.4f}")
    print(f"Min PCC    : {gene_df['pcc'].min():.4f}")

    return gene_df

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    dataset = torch.load(args.dataset_path)
    features = dataset["test"]["features"]
    labels = dataset["test"]["labels"]
    gene_names = dataset["gene_names"]
    sample_ids = dataset["test"]["sample_ids"]

    test_dataset = TensorDataset(features, labels)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = GeneExpressionModelHead(input_dim=1024, hidden_dim=512, output_dim=len(gene_names), num_heads=6)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    test_metrics = test_model(model, dataloader, device)

    preds, labels = output_predictions(model, dataloader, device)
    df_true = pd.DataFrame(labels, index=sample_ids, columns=gene_names)
    df_pred = pd.DataFrame(preds, index=sample_ids, columns=gene_names)

    df_true.to_csv(os.path.join(args.results_dir, "test_true_labels.csv"))
    df_pred.to_csv(os.path.join(args.results_dir, "test_pred_labels.csv"))

    gene_metrics = compute_genewise_pcc(df_true, df_pred)
    gene_metrics.to_csv(os.path.join(args.results_dir, "test_gene_metrics.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to test dataset (.pt)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights (.pth)")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    main(args)
