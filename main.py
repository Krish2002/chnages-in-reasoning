import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import os
import json
import argparse
from datetime import datetime

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer from HuggingFace model name or local path."""
    print(f"Using device: {device}\n")
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_stats_for_generation(model, tokenizer, generated_ids, prompt_length):
    """Calculate token-level statistics for a generation."""
    device = model.device
    inputs = {"input_ids": generated_ids.to(device)}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
        logits = outputs.logits

    stats = []
    for j in range(prompt_length - 1, generated_ids.shape[1] - 1):
        pos_logits = logits[0, j, :]
        entropy = torch.distributions.Categorical(logits=pos_logits).entropy().item()
        actual_next_token_id = generated_ids[0, j + 1]
        probabilities = F.softmax(pos_logits, dim=-1)
        prob_of_chosen_token = probabilities[actual_next_token_id].item()
        current_token_str = tokenizer.decode(generated_ids[0, j])
        next_token_str = tokenizer.decode(actual_next_token_id)
        stats.append({
            'current_token': current_token_str,
            'next_token': next_token_str,
            'entropy': entropy,
            'probability_of_next_token': prob_of_chosen_token,
        })
        
    return pd.DataFrame(stats)

def process_generations(model, tokenizer, prompts, max_new_tokens=256):
    """Process generations for all prompts and collect statistics."""
    device = model.device
    all_outputs = []

    for i, prompt in enumerate(prompts):
        print(f"\n\n{'=' * 35} PROMPT {i + 1}/{len(prompts)} {'=' * 35}")
        formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        prompt_length = inputs.input_ids.shape[1]
        print(f"PROMPT: \"{formatted_prompt[:500]}...\"\n")
        print("-" * 30)
        print("Generating...")
        
        generated_ids = model.generate(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        
        stats = get_stats_for_generation(model, tokenizer, generated_ids, prompt_length)
        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("Done.")
        print("-" * 30)
        
        all_outputs.append({
            "prompt_id": i + 1,
            "prompt": formatted_prompt,
            "stats": stats,
            "full_text": full_text,
        })
    return all_outputs

def load_prompts_from_jsonl(file_path):
    """Load prompts from a JSONL file, extracting only system and user roles."""
    prompts = []
    
    print(f"Loading prompts from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                conversations = data.get('conversations', [])
                
                # Extract only system and user messages
                system_user_messages = []
                for msg in conversations:
                    if msg.get('role') in ['system', 'user']:
                        system_user_messages.append({
                            'role': msg['role'],
                            'content': msg['content']
                        })
                
                if system_user_messages:
                    prompts.append(system_user_messages)
                else:
                    print(f"Warning: No system/user messages found in line {line_num}")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    print(f"Loaded {len(prompts)} prompts from {file_path}")
    return prompts

def analyze_token_qcuts(df, metric_column, model_name, n_qcuts=10):
    """Analyze token quantiles based on a metric."""
    print(f"\n{'=' * 30}")
    print(f"Token qcuts for {model_name} based on '{metric_column}'")
    print("=" * 30)
    
    qcut_col_name = f'{metric_column}_qcut'
    df[qcut_col_name] = pd.qcut(df[metric_column], n_qcuts, labels=False, duplicates='drop')
    qcut_intervals = pd.qcut(df[metric_column], n_qcuts, duplicates='drop')
    interval_map = {i: interval for i, interval in enumerate(qcut_intervals.cat.categories)}
    
    num_actual_qcuts = df[qcut_col_name].nunique()
    qcuts_to_display = sorted(list(set([0, num_actual_qcuts // 4, num_actual_qcuts // 2, num_actual_qcuts * 3 // 4, num_actual_qcuts - 1])))

    for qcut_idx in qcuts_to_display:
        df_qcut = df[df[qcut_col_name] == qcut_idx].copy()
        df_qcut.sort_values(by=metric_column, inplace=True)
        print(f"\n--- qcut {qcut_idx + 1}/{num_actual_qcuts} (range: {interval_map[qcut_idx]}) ---")
        num_samples = min(5, len(df_qcut))
        step = max(1, len(df_qcut) // num_samples)
        
        print(df_qcut[['current_token', 'next_token', metric_column]].iloc[::step])
    
    return df

def process_single_model(model_name, prompts, max_new_tokens, output_dir, timestamp):
    """Process a single model and save its results."""
    print(f"\n{'#' * 60}")
    print(f"PROCESSING MODEL: {model_name}")
    print(f"{'#' * 60}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Process all generations
    print(f"\nProcessing {len(prompts)} prompts...")
    all_results = process_generations(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
    
    # Create comprehensive dataframe with all statistics
    all_stats_data = []
    
    for result in all_results:
        prompt_id = result['prompt_id']
        stats_df = result['stats']
        full_text = result['full_text']
        
        # Add prompt information to each row
        for idx, row in stats_df.iterrows():
            stats_data = {
                'prompt_id': prompt_id,
                'full_text': full_text,
                'token_position': idx,
                'current_token': row['current_token'],
                'next_token': row['next_token'],
                'entropy': row['entropy'],
                'probability_of_next_token': row['probability_of_next_token'],
            }
            all_stats_data.append(stats_data)
    
    # Create final dataframe
    final_df = pd.DataFrame(all_stats_data)
    
    # Get model name for file naming (extract last part of path)
    model_name_clean = model_name.split("/")[-1].replace("-", "_").replace(".", "_")
    
    # Print summary statistics
    print(f"\n{'#' * 25} Analysis Summary for {model_name_clean} {'#' * 25}")
    print(f"Total tokens analyzed: {len(final_df)}")
    print(f"Total prompts processed: {len(prompts)}")
    print(f"Model used: {model_name}")
    
    print(f"\nEntropy statistics:")
    print(final_df['entropy'].describe())
    
    print(f"\nProbability of chosen token statistics:")
    print(final_df['probability_of_next_token'].describe())
    
    # Analyze token quantiles and add to dataframe
    final_df = analyze_token_qcuts(final_df, 'entropy', model_name_clean, n_qcuts=10)
    
    # Save the dataframe
    output_filename = os.path.join(output_dir, f'{model_name_clean}_stats_{timestamp}.csv')
    final_df.to_csv(output_filename, index=False)
    print(f"\nDataframe saved to: {output_filename}")
    
    # Save additional summary statistics
    summary_stats = {
        'model_name': model_name,
        'model_name_clean': model_name_clean,
        'total_tokens': len(final_df),
        'total_prompts': len(prompts),
        'mean_entropy': final_df['entropy'].mean(),
        'std_entropy': final_df['entropy'].std(),
        'mean_probability': final_df['probability_of_next_token'].mean(),
        'std_probability': final_df['probability_of_next_token'].std(),
        'timestamp': timestamp
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_filename = os.path.join(output_dir, f'{model_name_clean}_summary_{timestamp}.csv')
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary statistics saved to: {summary_filename}")
    
    # Save interesting findings to separate files
    high_entropy_tokens = final_df.nlargest(100, 'entropy')[['current_token', 'next_token', 'entropy', 'prompt_id', 'token_position']]
    low_entropy_tokens = final_df.nsmallest(100, 'entropy')[['current_token', 'next_token', 'entropy', 'prompt_id', 'token_position']]
    confident_tokens = final_df.nlargest(100, 'probability_of_next_token')[['current_token', 'next_token', 'probability_of_next_token', 'prompt_id', 'token_position']]
    
    high_entropy_filename = os.path.join(output_dir, f'{model_name_clean}_high_entropy_tokens_{timestamp}.csv')
    low_entropy_filename = os.path.join(output_dir, f'{model_name_clean}_low_entropy_tokens_{timestamp}.csv')
    confident_tokens_filename = os.path.join(output_dir, f'{model_name_clean}_confident_tokens_{timestamp}.csv')
    
    high_entropy_tokens.to_csv(high_entropy_filename, index=False)
    low_entropy_tokens.to_csv(low_entropy_filename, index=False)
    confident_tokens.to_csv(confident_tokens_filename, index=False)
    
    print(f"High entropy tokens saved to: {high_entropy_filename}")
    print(f"Low entropy tokens saved to: {low_entropy_filename}")
    print(f"Confident tokens saved to: {confident_tokens_filename}")
    
    # Clear model from memory
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'model_name': model_name,
        'model_name_clean': model_name_clean,
        'stats_file': output_filename,
        'summary_file': summary_filename,
        'total_tokens': len(final_df),
        'mean_entropy': final_df['entropy'].mean(),
        'mean_probability': final_df['probability_of_next_token'].mean()
    }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze multiple language models for token-level statistics')
    parser.add_argument('--models', nargs='+', required=True, 
                       help='List of model paths or HuggingFace model names (space-separated)')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--output-dir', type=str, 
                       default="/tmpdir/m24047krsh/changes_in_reasoning/results",
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Configuration
    MODEL_NAMES = args.models
    MAX_NEW_TOKENS = args.max_new_tokens
    DATA_FILE = "data_full.jsonl"  # Explicitly use data_full.jsonl
    OUTPUT_DIR = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load prompts from JSONL file
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found!")
        return
    
    PROMPTS = load_prompts_from_jsonl(DATA_FILE)
    
    if not PROMPTS:
        print("No prompts loaded! Exiting.")
        return
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each model
    all_model_results = []
    
    for i, model_name in enumerate(MODEL_NAMES, 1):
        print(f"\n{'=' * 80}")
        print(f"PROCESSING MODEL {i}/{len(MODEL_NAMES)}: {model_name}")
        print(f"{'=' * 80}")
        
        try:
            result = process_single_model(model_name, PROMPTS, MAX_NEW_TOKENS, OUTPUT_DIR, timestamp)
            all_model_results.append(result)
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue
    
    # Create overall summary
    if all_model_results:
        print(f"\n{'#' * 60}")
        print("OVERALL SUMMARY")
        print(f"{'#' * 60}")
        
        summary_data = []
        for result in all_model_results:
            summary_data.append({
                'model_name': result['model_name'],
                'model_name_clean': result['model_name_clean'],
                'total_tokens': result['total_tokens'],
                'mean_entropy': result['mean_entropy'],
                'mean_probability': result['mean_probability'],
                'stats_file': result['stats_file'],
                'summary_file': result['summary_file']
            })
        
        overall_summary_df = pd.DataFrame(summary_data)
        overall_summary_filename = os.path.join(OUTPUT_DIR, f'overall_summary_{timestamp}.csv')
        overall_summary_df.to_csv(overall_summary_filename, index=False)
        
        print(f"\nOverall summary saved to: {overall_summary_filename}")
        print(f"\nProcessed {len(all_model_results)} models successfully:")
        for result in all_model_results:
            print(f"  - {result['model_name_clean']}: {result['total_tokens']} tokens, "
                  f"mean entropy: {result['mean_entropy']:.4f}, "
                  f"mean probability: {result['mean_probability']:.4f}")
    
    print(f"\nAnalysis complete! All data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
