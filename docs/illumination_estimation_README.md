# Illumination Estimation Pipeline

This pipeline tests the hypothesis that VLM embeddings from both LLM and visual layers can be used for illumination estimation. It extracts embeddings from images, trains linear regression models, and evaluates performance using angular error metrics.

## Pipeline Components

### 1. Embedding Extraction
- **Script**: `scripts/extract_illumination_embeddings.py`
- **Purpose**: Extract VLM embeddings from `.JPG` images in `PNG/` subfolder
- **Output**: Embeddings saved to `data/embeddings/qwen2.5_7B/{experiment_name}/`

### 2. Linear Regression Training
- **Script**: `utils/cube++/train_illumination_regression.py`
- **Purpose**: Train regression models on different embedding types
- **Embedding Types**: `vision_pooled_mean`, `lm_pooled_mean`, `projected_pooled_mean`
- **Output**: Results JSON with train/test metrics

### 3. Experiment Runner
- **Script**: `scripts/run_illumination_experiments.sh`
- **Purpose**: Run complete pipeline for multiple configurations
- **Output**: Combined results in `results/illumination/`

## Data Structure

Each dataset should have this structure:
```
dataset_path/
├── gt.csv          # Ground truth illumination data
└── PNG/            # Images folder
    ├── 00_0010.JPG
    ├── 00_0011.JPG
    └── ...
```

### Ground Truth CSV Format
`gt.csv` should contain:
```csv
image,mean_r,mean_g,mean_b
00_0010,0.3864471120297919,0.4596439138217117,0.15390897414849627
00_0011,0.4123456789,0.3987654321,0.1888888889
...
```

## Configuration Files

Create JSON configs in `configs/` directory:

```json
{
  "experiment_name": "illumination_hypothesis_test",
  "train_data_path": "path/to/your/train_dataset",
  "test_data_path": "path/to/your/test_dataset", 
  "outdir_root": "data/embeddings/qwen2.5_7B",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "prompt": "What is the illumination in this image?",
  "init_prompt": "You are an expert at analyzing illumination in images.",
  "device": "cuda:0",
  "save_tokens": false,
  "restart_model_per_image": true
}
```

## Usage

### Quick Start

1. **Create your config file**:
   ```bash
   cp configs/illumination_hypothesis_test.json configs/my_experiment.json
   # Edit paths in my_experiment.json
   ```

2. **Run the complete pipeline**:
   ```bash
   ./scripts/run_illumination_experiments.sh configs/my_experiment.json
   ```

3. **Check results**:
   ```bash
   cat results/illumination/my_experiment_results.json
   ```

### Manual Steps

If you prefer to run steps manually:

1. **Extract embeddings for train data**:
   ```bash
   python scripts/extract_illumination_embeddings.py --config configs/train_config.json
   ```

2. **Extract embeddings for test data**:
   ```bash
   python scripts/extract_illumination_embeddings.py --config configs/test_config.json
   ```

3. **Train regression models**:
   ```bash
   python utils/cube++/train_illumination_regression.py --config configs/experiment_config.json
   ```

### Multiple Experiments

Run multiple experiments at once:
```bash
./scripts/run_illumination_experiments.sh configs/exp1.json configs/exp2.json configs/exp3.json
```

## Results Format

Results are saved as JSON in `results/illumination/{experiment_name}_results.json`:

```json
{
  "experiment_name": "illumination_hypothesis_test",
  "config": {...},
  "results": {
    "vision_pooled_mean": {
      "train_metrics": {"mean_repr_ang_error": 12.34, "mse": 0.0456},
      "test_metrics": {"mean_repr_ang_error": 15.67, "mse": 0.0589},
      "train_samples": 1000,
      "test_samples": 200,
      "embedding_dim": 1536
    },
    "lm_pooled_mean": {
      "train_metrics": {"mean_repr_ang_error": 10.23, "mse": 0.0321},
      "test_metrics": {"mean_repr_ang_error": 13.45, "mse": 0.0432},
      "train_samples": 1000,
      "test_samples": 200,
      "embedding_dim": 3584
    },
    "projected_pooled_mean": {
      "train_metrics": {"mean_repr_ang_error": 11.56, "mse": 0.0387},
      "test_metrics": {"mean_repr_ang_error": 14.78, "mse": 0.0512},
      "train_samples": 1000,
      "test_samples": 200,
      "embedding_dim": 3584
    }
  }
}
```

## Metrics

- **Angular Error**: Reproduction angular error in degrees (lower is better)
- **MSE**: Mean squared error between predicted and ground truth RGB values
- **Train/Test**: Separate metrics for training and test sets

## Hypothesis Testing

The pipeline allows you to test:
1. **Visual vs LLM embeddings**: Compare `vision_pooled_mean` vs `lm_pooled_mean`
2. **Projected embeddings**: Test if `projected_pooled_mean` (vision→LLM space) performs better
3. **Generalization**: Compare train vs test performance to assess overfitting

## Dependencies

- Python 3.8+
- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- tqdm

## Troubleshooting

1. **CUDA out of memory**: Set `restart_model_per_image: true` in config
2. **Missing images**: Check that all `.JPG` files exist in `PNG/` subfolder
3. **CSV format errors**: Ensure `gt.csv` has correct column names and no missing values
4. **Path issues**: Use absolute paths or ensure relative paths are correct from project root