"""
Train linear regression models for illumination estimation using VLM embeddings.
Supports training on different embedding types: vision_pooled_mean, lm_pooled_mean, projected_pooled_mean.

Usage:
    python utils/cube++/train_illumination_regression.py --config path/to/config.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import sys
import os

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from .calc_metrics import calc_metrics, parse_csv


def load_embeddings_and_gt(embeddings_dir: Path, gt_csv_path: Path):
    """
    Load embeddings and ground truth illumination data.
    
    Args:
        embeddings_dir: Directory containing embedding subfolders
        gt_csv_path: Path to gt.csv file with illumination data
        
    Returns:
        embeddings_dict: Dict mapping embedding_type -> (image_names, embeddings)
        gt_df: DataFrame with ground truth illumination
    """
    # Load ground truth
    gt_df = pd.read_csv(gt_csv_path)
    gt_df = gt_df.set_index('image')
    
    # Find all embedding directories
    embedding_dirs = [d for d in embeddings_dir.iterdir() if d.is_dir()]
    image_names = sorted([d.stem for d in embedding_dirs])
    
    # Load different embedding types
    embedding_types = ['vision_pooled_mean', 'lm_pooled_mean', 'projected_pooled_mean']
    embeddings_dict = {}
    
    for emb_type in embedding_types:
        embeddings = []
        valid_images = []
        
        for img_name in image_names:
            emb_path = embeddings_dir / img_name / f"{emb_type}.npy"
            if emb_path.exists():
                emb = np.load(emb_path)
                embeddings.append(emb)
                valid_images.append(img_name)
        
        if embeddings:
            embeddings_dict[emb_type] = (valid_images, np.array(embeddings))
            print(f"Loaded {len(embeddings)} {emb_type} embeddings")
    
    return embeddings_dict, gt_df, image_names


def train_regression_model(X_train: np.typing.NDArray, y_train, X_test: np.typing.NDArray, y_test):
    """
    Train linear regression model and calculate metrics.
    
    Args:
        X_train: Training embeddings
        y_train: Training ground truth
        X_test: Test embeddings  
        y_test: Test ground truth
        
    Returns:
        model: Trained regression model
        train_pred: Training predictions
        test_pred: Test predictions
        train_metrics: Training metrics
        test_metrics: Test metrics
    """
    # Train model
    model = Ridge()
    X_train = X_train.squeeze()
    X_test = X_test.squeeze()
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate MSE metrics
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    # Prepare data for angular error calculation
    train_gt_df = pd.DataFrame({
        'image': [f"img_{i}" for i in range(len(y_train))],
        'r': y_train[:, 0],
        'g': y_train[:, 1], 
        'b': y_train[:, 2]
    })
    
    train_pred_df = pd.DataFrame({
        'image': [f"img_{i}" for i in range(len(train_pred))],
        'r': train_pred[:, 0],
        'g': train_pred[:, 1],
        'b': train_pred[:, 2]
    })
    
    test_gt_df = pd.DataFrame({
        'image': [f"img_{i}" for i in range(len(y_test))],
        'r': y_test[:, 0],
        'g': y_test[:, 1],
        'b': y_test[:, 2]
    })
    
    test_pred_df = pd.DataFrame({
        'image': [f"img_{i}" for i in range(len(test_pred))],
        'r': test_pred[:, 0],
        'g': test_pred[:, 1],
        'b': test_pred[:, 2]
    })
    
    # Calculate angular errors
    train_metrics = calc_metrics(train_gt_df, train_pred_df, 'indoor')
    test_metrics = calc_metrics(test_gt_df, test_pred_df, 'indoor')
    
    # Add MSE to metrics
    train_metrics['mse'] = train_mse
    test_metrics['mse'] = test_mse
    
    return model, train_pred, test_pred, train_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train linear regression models for illumination estimation.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config file with experiment settings.")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = json.load(f)
    
    print("Config:", cfg)
    
    experiment_name = cfg.get("experiment_name")
    if not experiment_name:
        raise SystemExit("Config must contain 'experiment_name' field.")
    
    # Paths
    embeddings_root = Path(cfg.get("outdir_root", "data/embeddings/qwen2.5_7B"))
    train_embeddings_dir = embeddings_root / f"{experiment_name}_train"
    test_embeddings_dir = embeddings_root / f"{experiment_name}_test"
    
    train_data_path = Path(cfg.get("train_data_path"))
    test_data_path = Path(cfg.get("test_data_path"))
    
    train_gt_csv = train_data_path / "gt.csv"
    test_gt_csv = test_data_path / "gt.csv"
    
    # Check paths exist
    if not train_embeddings_dir.exists():
        raise SystemExit(f"Train embeddings directory not found: {train_embeddings_dir}")
    if not test_embeddings_dir.exists():
        raise SystemExit(f"Test embeddings directory not found: {test_embeddings_dir}")
    if not train_gt_csv.exists():
        raise SystemExit(f"Train GT CSV not found: {train_gt_csv}")
    if not test_gt_csv.exists():
        raise SystemExit(f"Test GT CSV not found: {test_gt_csv}")
    
    # Load data
    print("Loading train embeddings and ground truth...")
    train_embeddings_dict, train_gt_df, train_image_names = load_embeddings_and_gt(
        train_embeddings_dir, train_gt_csv)
    print("Loading test embeddings and ground truth...")
    test_embeddings_dict, test_gt_df, test_image_names = load_embeddings_and_gt(
        test_embeddings_dir, test_gt_csv)
    
    # Prepare ground truth arrays
    train_gt_arrays = {}
    test_gt_arrays = {}
    for emb_type, (valid_images, _) in train_embeddings_dict.items():
        # Filter ground truth for valid images
        train_gt = []
        for img_name in valid_images:
            if img_name in train_gt_df.index:
                train_gt.append([train_gt_df.loc[img_name, 'mean_r'],
                               train_gt_df.loc[img_name, 'mean_g'], 
                               train_gt_df.loc[img_name, 'mean_b']])
        train_gt_arrays[emb_type] = np.array(train_gt)
    for emb_type, (valid_images, _) in test_embeddings_dict.items():
        # Filter ground truth for valid images
        test_gt = []
        for img_name in valid_images:
            if img_name in test_gt_df.index:
                test_gt.append([test_gt_df.loc[img_name, 'mean_r'],
                              test_gt_df.loc[img_name, 'mean_g'],
                              test_gt_df.loc[img_name, 'mean_b']])
        test_gt_arrays[emb_type] = np.array(test_gt)
    
    # Train models for each embedding type
    results = {
        "experiment_name": experiment_name,
        "config": cfg,
        "results": {}
    }
    
    for emb_type in train_embeddings_dict.keys():
        print(f"\nTraining model for {emb_type}...")
        
        # Get data
        _, X_train = train_embeddings_dict[emb_type]
        _, X_test = test_embeddings_dict[emb_type]
        y_train = train_gt_arrays[emb_type]
        y_test = test_gt_arrays[emb_type]
        
        # Train model
        model, train_pred, test_pred, train_metrics, test_metrics = train_regression_model(
            X_train, y_train, X_test, y_test)
        
        # Store results
        results["results"][emb_type] = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "embedding_dim": X_train.shape[1]
        }
        
        print(f"  Train angle: {train_metrics['mean_repr_ang_error']:.2f}°")
        print(f"  Test angle: {test_metrics['mean_repr_ang_error']:.2f}°")
    
    # Save results
    results_dir = Path("results/illumination")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"{experiment_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("\nSummary:")
    for emb_type, res in results["results"].items():
        print(f"  {emb_type}:")
        print(f"    Train angle: {res['train_metrics']['mean_repr_ang_error']:.2f}°")
        print(f"    Test angle: {res['test_metrics']['mean_repr_ang_error']:.2f}°")


if __name__ == "__main__":
    main()