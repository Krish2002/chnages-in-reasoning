#!/usr/bin/env python3
"""
Example usage of the model comparison analysis script.
This demonstrates how to use the functions programmatically instead of via command line.
"""

import sys
from pathlib import Path
from model_comparison_analysis import (
    load_model_and_tokenizer,
    load_prompts_from_jsonl,
    process_generations,
    plot_entropy_comparison,
    analyze_token_qcuts
)

def main():
    # Configuration - modify these paths to match your setup
    MODEL_A_PATH = "/path/to/your/first/model/checkpoint"
    MODEL_B_PATH = "/path/to/your/second/model/checkpoint"
    DATA_PATH = "data_full.jsonl"
    NUM_EXAMPLES = 100  # Start with a small number for testing
    MAX_NEW_TOKENS = 256
    OUTPUT_DIR = "example_results"
    
    print("Model Comparison Analysis - Example Usage")
    print("=" * 50)
    
    # Check if data file exists
    if not Path(DATA_PATH).exists():
        print(f"Error: Data file {DATA_PATH} not found!")
        print("Please make sure the data_full.jsonl file is in the current directory.")
        return
    
    # Load models
    print("Loading models...")
    try:
        model_a, tokenizer = load_model_and_tokenizer(MODEL_A_PATH)
        model_b, _ = load_model_and_tokenizer(MODEL_B_PATH)
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please check your model paths and make sure they exist.")
        return
    
    # Load prompts from JSONL
    print("Loading prompts from JSONL file...")
    try:
        prompts = load_prompts_from_jsonl(DATA_PATH, NUM_EXAMPLES)
        if len(prompts) == 0:
            print("No valid prompts found in the data file!")
            return
        print(f"✓ Loaded {len(prompts)} prompts")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return
    
    # Process generations
    print("Processing generations...")
    try:
        all_results = process_generations(model_a, model_b, tokenizer, prompts, MAX_NEW_TOKENS)
        print(f"✓ Processed {len(all_results)} examples")
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    
    # Analyze and plot results
    print("Analyzing results...")
    try:
        agg_df_a, agg_df_b = plot_entropy_comparison(all_results, OUTPUT_DIR)
        print("✓ Analysis complete")
    except Exception as e:
        print(f"Error during analysis: {e}")
        return
    
    # Analyze token quantiles
    print("Analyzing token quantiles...")
    try:
        analyze_token_qcuts(agg_df_a, 'entropy', 'Model A - aggregate', n_qcuts=10)
        analyze_token_qcuts(agg_df_b, 'entropy', 'Model B - aggregate', n_qcuts=10)
        print("✓ Token analysis complete")
    except Exception as e:
        print(f"Error during token analysis: {e}")
        return
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Examples processed: {len(prompts)}")
    print(f"Model A total tokens: {len(agg_df_a)}")
    print(f"Model B total tokens: {len(agg_df_b)}")
    print(f"Model A mean entropy: {agg_df_a['entropy'].mean():.4f}")
    print(f"Model B mean entropy: {agg_df_b['entropy'].mean():.4f}")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("=" * 50)

if __name__ == "__main__":
    main() 