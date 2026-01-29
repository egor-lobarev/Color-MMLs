#!/bin/bash

# Illumination Estimation Experiments Runner
# Usage: ./scripts/run_illumination_experiments.sh config1.json config2.json ...

set -e  # Exit on any error

if [ $# -eq 0 ]; then
    echo "Usage: $0 config1.json config2.json ..."
    echo "Example: $0 configs/illumination_experiment1.json configs/illumination_experiment2.json"
    exit 1
fi

echo "Starting illumination estimation experiments..."
echo "=============================================="

# Create results directory
mkdir -p results/illumination

# Process each config file
for config_file in "$@"; do
    if [ ! -f "$config_file" ]; then
        echo "Error: Config file not found: $config_file"
        continue
    fi
    
    echo ""
    echo "Processing config: $config_file"
    echo "----------------------------------------"
    
    # Load experiment name from config
    experiment_name=$(python3 -c "
import json
with open('$config_file', 'r') as f:
    config = json.load(f)
print(config.get('experiment_name', 'unknown'))
")
    
    echo "Experiment name: $experiment_name"
    
    # Create modified configs for train and test data
    train_config="configs/${experiment_name}_train_temp.json"
    test_config="configs/${experiment_name}_test_temp.json"
    
    # Modify config for train data
    python3 -c "
import json
with open('$config_file', 'r') as f:
    config = json.load(f)

# Create train config
train_config = config.copy()
train_config['experiment_name'] = config['experiment_name'] + '_train'
train_config['data_path'] = config['train_data_path']

with open('$train_config', 'w') as f:
    json.dump(train_config, f, indent=2)

# Create test config  
test_config = config.copy()
test_config['experiment_name'] = config['experiment_name'] + '_test'
test_config['data_path'] = config['test_data_path']

with open('$test_config', 'w') as f:
    json.dump(test_config, f, indent=2)
"
    
    echo "Extracting embeddings for training data..."
    python3 scripts/extract_illumination_embeddings.py --config "$train_config"
    
    echo "Extracting embeddings for test data..."
    python3 scripts/extract_illumination_embeddings.py --config "$test_config"
    
    # Clean up temporary configs
    rm "$train_config" "$test_config"
    
    echo "Training regression models..."
    python3 utils/cube++/train_illumination_regression.py --config "$config_file"
    
    echo "Experiment completed for: $experiment_name"
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved in: results/illumination/"

# Show summary of all results
echo ""
echo "Results Summary:"
echo "---------------"
for result_file in results/illumination/*_results.json; do
    if [ -f "$result_file" ]; then
        experiment=$(basename "$result_file" _results.json)
        echo ""
        echo "Experiment: $experiment"
        python3 -c "
import json
with open('$result_file', 'r') as f:
    results = json.load(f)

for emb_type, res in results['results'].items():
    train_angle = res['train_metrics']['mean_repr_ang_error']
    test_angle = res['test_metrics']['mean_repr_ang_error']
    print(f'  {emb_type}: Train={train_angle:.2f}°, Test={test_angle:.2f}°')
"
    fi
done