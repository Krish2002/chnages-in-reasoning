# Model Comparison Analysis for Reasoning Tasks

This script analyzes and compares multiple language model checkpoints by examining their token-level entropy and generation statistics. It's specifically designed for analyzing how model behavior changes during training, particularly for reasoning tasks.

## 🎯 Purpose

The script helps researchers understand:
- How entropy distributions change across different training checkpoints
- Which tokens have high/low entropy during generation
- How model confidence evolves during training
- Differences in generation patterns between checkpoints

## 🚀 Quick Start

### Basic Usage (2 Models)
```bash
python model_comparison_analysis.py \
    --model_paths /path/to/model1 /path/to/model2 \
    --model_names "Base" "Finetuned"
```

### Multi-Model Comparison (4 Checkpoints)
```bash
python model_comparison_analysis.py \
    --model_paths /tmpdir/m24047krsh/models/qwen-2.5-1.5b-instruct-reasoning-sft-checkpoint-1-of-10 \
                  /tmpdir/m24047krsh/models/qwen-2.5-1.5b-instruct-reasoning-sft-checkpoint-2-of-10 \
                  /tmpdir/m24047krsh/models/qwen-2.5-1.5b-instruct-reasoning-sft-checkpoint-4-of-10 \
                  /tmpdir/m24047krsh/models/qwen-2.5-1.5b-instruct-reasoning-sft-checkpoint-10-of-10 \
    --model_names "Checkpoint_1" "Checkpoint_2" "Checkpoint_4" "Checkpoint_10" \
    --num_examples 1000 \
    --max_new_tokens 256
```

## 📋 Requirements

Install the required dependencies:
```bash
pip install torch transformers numpy pandas matplotlib
```

## 🔧 Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model_paths` | str | ✅ | - | Paths to model checkpoints (space-separated) |
| `--model_names` | str | ✅ | - | Names for the models (space-separated, same order as paths) |
| `--data_path` | str | ❌ | `/tmpdir/m24047krsh/changes_in_reasoning/data_full.jsonl` | Path to JSONL data file |
| `--num_examples` | int | ❌ | 1000 | Number of examples to process |
| `--max_new_tokens` | int | ❌ | 256 | Maximum new tokens to generate |
| `--output_dir` | str | ❌ | `/tmpdir/m24047krsh/changes_in_reasoning/results` | Output directory for results |
| `--save_detailed_tokens` | flag | ❌ | False | Save detailed token-level stats (can be large) |

## 📊 Data Format

The script expects a JSONL file where each line contains:
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a helpful assistant. Think step by step..."
    },
    {
      "role": "user", 
      "content": "What is the value of x in the equation..."
    },
    {
      "role": "assistant",
      "content": "<think>...</think>\n\nAnswer: \\boxed{...}"
    }
  ]
}
```

The script extracts the system and user messages to create prompts for generation.

## 📈 Output Files

### Core Analysis Files
- **`{model_name}_token_stats.csv`** - Aggregated token statistics for each model
- **`prompt_level_results.csv`** - Summary statistics for each prompt across all models
- **`results_summary.json`** - Overall statistics and model paths

### Visualizations
- **`aggregated_entropy_distribution.png`** - Histogram comparing entropy distributions
- **`entropy_over_steps.png`** - Entropy over generation steps for sample prompts

### Detailed Analysis (Optional)
- **`{model_name}_all_tokens_detailed.csv`** - Detailed token stats with prompt tracking (only with `--save_detailed_tokens`)

### Example Output Structure
```
results/
├── Checkpoint_1_token_stats.csv
├── Checkpoint_2_token_stats.csv
├── Checkpoint_4_token_stats.csv
├── Checkpoint_10_token_stats.csv
├── prompt_level_results.csv
├── aggregated_entropy_distribution.png
├── entropy_over_steps.png
├── results_summary.json
└── [detailed files if --save_detailed_tokens used]
```

## 🔍 What Gets Analyzed

### Token-Level Statistics
- **Entropy**: Uncertainty in next-token prediction
- **Probability of chosen token**: How confident the model was
- **Current/Next token**: The actual tokens generated
- **Generation step**: Which step in the generation process

### Key Metrics
- **Mean/Median/Std entropy**: Overall uncertainty patterns
- **Min/Max entropy**: Extremes in model confidence
- **Token quantiles**: High/low entropy token analysis

## 🧪 Testing and Validation

### Quick Test (5 examples)
```bash
python model_comparison_analysis.py \
    --model_paths /tmpdir/m24047krsh/models/qwen-2.5-1.5b-instruct-reasoning-sft-checkpoint-1-of-10 \
                  /tmpdir/m24047krsh/models/qwen-2.5-1.5b-instruct-reasoning-sft-checkpoint-2-of-10 \
    --model_names "Checkpoint_1" "Checkpoint_2" \
    --num_examples 5 \
    --max_new_tokens 50
```

### Full Analysis with Detailed Stats
```bash
python model_comparison_analysis.py \
    --model_paths /tmpdir/m24047krsh/models/qwen-2.5-1.5b-instruct-reasoning-sft-checkpoint-1-of-10 \
                  /tmpdir/m24047krsh/models/qwen-2.5-1.5b-instruct-reasoning-sft-checkpoint-2-of-10 \
                  /tmpdir/m24047krsh/models/qwen-2.5-1.5b-instruct-reasoning-sft-checkpoint-4-of-10 \
                  /tmpdir/m24047krsh/models/qwen-2.5-1.5b-instruct-reasoning-sft-checkpoint-10-of-10 \
    --model_names "Checkpoint_1" "Checkpoint_2" "Checkpoint_4" "Checkpoint_10" \
    --num_examples 1000 \
    --max_new_tokens 256 \
    --save_detailed_tokens
```

## 🔬 Understanding the Results

### High Entropy Tokens
- Indicate model uncertainty
- Often occur at decision points
- May correspond to "thinking" moments

### Low Entropy Tokens
- Indicate high confidence
- Often punctuation, common words, or formulaic responses
- May show learned patterns

### Entropy Trends
- **Decreasing entropy**: Model becomes more confident during generation
- **Spikes in entropy**: Model faces difficult decisions
- **Checkpoint differences**: Training progress indicators



## 📄 License

This script is provided for research purposes. Please cite appropriately if used in academic work. 
