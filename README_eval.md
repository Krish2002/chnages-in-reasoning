# Model Evaluation Script

This script carries out experiments to compare multiple pre and post finetuning models by analyzing token entropy and probability distributions during text generation.

## Features

- **Multi-Model Comparison**: Compare entropy and probability distributions between 4 models during text generation
- **Teacher Forcing**: Option to analyze models on ground truth sequences
- **Comprehensive Data Export**: Saves all results in structured format for later analysis and plotting
- **Quantile Analysis**: Analyzes token distributions across different entropy quantiles for each model
- **Token-level Statistics**: Detailed statistics for each token including current token, next token, entropy, and probability
- **Model Comparison Summary**: Direct comparison of key metrics across all models

## Installation

```bash
pip install -r requirements_eval.txt
```

## Usage

### Basic Usage

```bash
python eval.py \
    --model_1 /path/to/model1 \
    --model_2 /path/to/model2 \
    --model_3 /path/to/model3 \
    --model_4 /path/to/model4
```

### Advanced Usage with Custom Names

```bash
python eval.py \
    --model_1 /path/to/base/model \
    --model_2 /path/to/checkpoint1 \
    --model_3 /path/to/checkpoint2 \
    --model_4 /path/to/final/model \
    --model_names base checkpoint1 checkpoint2 final \
    --output_dir ./results \
    --max_new_tokens 4096 \
    --experiment_name "training_progression" \
    --teacher_forced
```

### Command Line Arguments

- `--model_1`: Path to Model 1
- `--model_2`: Path to Model 2  
- `--model_3`: Path to Model 3
- `--model_4`: Path to Model 4
- `--model_names`: Custom names for the models (4 space-separated names, optional)
- `--output_dir`: Output directory for results (default: `./results`)
- `--max_new_tokens`: Maximum new tokens to generate (default: 4096)
- `--teacher_forced`: Use teacher forcing instead of generation
- `--experiment_name`: Name for this experiment (default: `model_comparison`)

## Output Structure

The script creates the following directory structure:

```
results/
├── prompt_1/
│   ├── model_1_stats_20241201_143022.csv
│   ├── model_2_stats_20241201_143022.csv
│   ├── model_3_stats_20241201_143022.csv
│   ├── model_4_stats_20241201_143022.csv
│   └── generated_texts_20241201_143022.json
├── prompt_2/
│   └── ...
├── aggregated_model_1_stats_20241201_143022.csv
├── aggregated_model_2_stats_20241201_143022.csv
├── aggregated_model_3_stats_20241201_143022.csv
├── aggregated_model_4_stats_20241201_143022.csv
├── model_1_token_analysis_current_20241201_143022.csv
├── model_2_token_analysis_current_20241201_143022.csv
├── model_3_token_analysis_current_20241201_143022.csv
├── model_4_token_analysis_current_20241201_143022.csv
├── model_1_token_analysis_next_20241201_143022.csv
├── model_2_token_analysis_next_20241201_143022.csv
├── model_3_token_analysis_next_20241201_143022.csv
├── model_4_token_analysis_next_20241201_143022.csv
├── summary_stats_20241201_143022.json
├── qcut_analysis_20241201_143022.json
└── model_comparison_20241201_143022.json
```

## Output Files

### Individual Prompt Results
- `model_X_stats_*.csv`: Token-level statistics for each model for each prompt
- `generated_texts_*.json`: Generated text outputs for all models

### Aggregated Results
- `aggregated_model_X_stats_*.csv`: All statistics combined for each model
- `summary_stats_*.json`: Summary statistics including means, standard deviations, etc. for all models

### Token Analysis
- `model_X_token_analysis_current_*.csv`: Statistics grouped by current token for each model
- `model_X_token_analysis_next_*.csv`: Statistics grouped by next token for each model

### Quantile Analysis
- `qcut_analysis_*.json`: Token analysis across entropy quantiles for all models

### Model Comparison
- `model_comparison_*.json`: Direct comparison of key metrics across all models

## Data Format

### CSV Files
Each CSV contains the following columns:
- `current_token`: The current token being analyzed
- `next_token`: The next token that was generated
- `entropy`: Entropy of the token distribution at that position
- `probability_of_next_token`: Probability assigned to the actual next token

### JSON Files
- `generated_texts_*.json`: Contains the full prompt and generated text for all models
- `summary_stats_*.json`: Contains aggregated statistics and metadata for all models
- `qcut_analysis_*.json`: Contains quantile analysis results for all models
- `model_comparison_*.json`: Contains direct comparison metrics across all models

## Example Analysis

After running the script, you can analyze the results using pandas:

```python
import pandas as pd
import json

# Load aggregated results for all models
models = ['model_1', 'model_2', 'model_3', 'model_4']
dfs = {}
for model in models:
    dfs[model] = pd.read_csv(f'results/aggregated_{model}_stats_20241201_143022.csv')

# Compare entropy distributions
print("Entropy statistics comparison:")
for model in models:
    print(f"\n{model}:")
    print(dfs[model]['entropy'].describe())

# Load model comparison summary
with open('results/model_comparison_20241201_143022.json', 'r') as f:
    comparison = json.load(f)

print("\nDirect model comparison:")
for model, stats in comparison['model_comparison'].items():
    print(f"\n{model}:")
    print(f"  Mean entropy: {stats['mean_entropy']:.4f}")
    print(f"  Mean probability: {stats['mean_probability']:.4f}")
    print(f"  95th percentile entropy: {stats['entropy_percentiles']['95%']:.4f}")

# Find high entropy tokens for each model
for model in models:
    high_entropy = dfs[model][dfs[model]['entropy'] > dfs[model]['entropy'].quantile(0.9)]
    print(f"\nHigh entropy tokens in {model}: {len(high_entropy)}")
```

## Example Use Cases

### Training Progression Analysis
Compare models at different training checkpoints to understand how entropy changes during training:

```bash
python eval.py \
    --model_1 /path/to/base/model \
    --model_2 /path/to/checkpoint_1000 \
    --model_3 /path/to/checkpoint_5000 \
    --model_4 /path/to/final_model \
    --model_names base checkpoint1k checkpoint5k final \
    --experiment_name "training_progression"
```

### Model Architecture Comparison
Compare different model variants or architectures:

```bash
python eval.py \
    --model_1 /path/to/model_1b \
    --model_2 /path/to/model_3b \
    --model_3 /path/to/model_7b \
    --model_4 /path/to/model_13b \
    --model_names 1B 3B 7B 13B \
    --experiment_name "scale_comparison"
```

### Fine-tuning Comparison
Compare different fine-tuning approaches:

```bash
python eval.py \
    --model_1 /path/to/base_model \
    --model_2 /path/to/sft_model \
    --model_3 /path/to/rlhf_model \
    --model_4 /path/to/dpo_model \
    --model_names base sft rlhf dpo \
    --experiment_name "finetuning_comparison"
```

## Notes

- The script uses the same prompts as the original notebook for reproducibility
- All results are timestamped to avoid overwriting previous experiments
- The script supports both local model paths and HuggingFace model names
- Teacher forcing mode is useful for analyzing model behavior on ground truth sequences
- Results are saved in a structured format that makes it easy to create custom visualizations
- The tokenizer from the first model is used for all models to ensure consistency
- The new `model_comparison_*.json` file provides a quick way to compare key metrics across all models 