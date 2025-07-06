# Model Token-Level Analysis

This script analyzes token-level statistics (entropy and probability) for language model generations. It's based on the notebook `sft-base-comparison.ipynb` but simplified to analyze a single model.

## Features

- **Token-level entropy analysis**: Calculates entropy for each token position during generation
- **Probability analysis**: Tracks the probability of chosen tokens
- **Comprehensive statistics**: Aggregates data across multiple prompts
- **Visualizations**: Creates plots showing distributions and relationships
- **Data export**: Saves all statistics to CSV files for further analysis

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Configure the model**: Edit the `MODEL_NAME` variable in `main.py` to point to your model:
   ```python
   MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Change this to your model
   ```

2. **Run the analysis**:
   ```bash
   python main.py
   ```

## Output Files

The script generates several output files with timestamps:

- `model_analysis_stats_YYYYMMDD_HHMMSS.csv`: Complete token-level statistics
- `model_analysis_summary_YYYYMMDD_HHMMSS.csv`: Summary statistics
- `model_analysis_plots_YYYYMMDD_HHMMSS.png`: Visualization plots

## Data Structure

The main CSV file contains the following columns:

- `prompt_id`: Identifier for each prompt
- `full_text`: Complete generated text
- `token_position`: Position of the token in the generation
- `current_token`: The current token being analyzed
- `next_token`: The next token that was generated
- `entropy`: Entropy of the token distribution at this position
- `probability_of_next_token`: Probability of the chosen next token

## Analysis Features

1. **Entropy Analysis**: Shows how uncertain the model is at each token position
2. **Probability Analysis**: Shows how confident the model is in its predictions
3. **Token Quantiles**: Groups tokens by entropy levels for detailed analysis
4. **Position Analysis**: Shows how entropy changes throughout the generation
5. **Interesting Findings**: Highlights highest/lowest entropy tokens and most confident predictions

## Customization

You can modify the script to:

- Change the number of prompts by editing the `PROMPTS` list
- Adjust the maximum generation length with `MAX_NEW_TOKENS`
- Add more analysis metrics in the `get_stats_for_generation` function
- Modify the visualizations in the plotting section

## Example Output

The script will print:
- Progress updates for each prompt
- Summary statistics
- Token quantile analysis
- Interesting findings (highest/lowest entropy tokens)
- File save confirmations

## Notes

- The script uses the same prompts as the original notebook for reproducibility
- All analysis is done on the generated tokens (not teacher-forced)
- The script automatically handles device selection (CPU/GPU)
- Timestamps are added to all output files to prevent overwrites 