import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import os
import json
import argparse
from datetime import datetime
from collections import defaultdict, Counter

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
        generation_length = generated_ids.shape[1] - prompt_length
        print("Done.")
        print("-" * 30)
        
        all_outputs.append({
            "prompt_id": i + 1,
            "prompt": formatted_prompt,
            "stats": stats,
            "full_text": full_text,
            "generation_length": generation_length,
        })
    return all_outputs

def load_prompts_from_jsonl(file_path, max_prompts=1000):
    """Load prompts from a JSONL file, extracting only system and user roles."""
    prompts = []
    
    print(f"Loading prompts from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if len(prompts) >= max_prompts:
                break
                
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

def analyze_token_entropy_changes(all_model_results, output_dir, timestamp):
    """Analyze how token entropies changed across models and which tokens changed the most."""
    print(f"\n{'#' * 60}")
    print("ANALYZING TOKEN ENTROPY CHANGES ACROSS MODELS")
    print(f"{'#' * 60}")
    
    # Collect all token data across models
    all_token_data = []
    model_names = []
    
    for result in all_model_results:
        model_name = result['model_name_clean']
        model_names.append(model_name)
        
        # Load the stats file
        stats_df = pd.read_csv(result['stats_file'])
        
        # Add model information
        stats_df['model'] = model_name
        all_token_data.append(stats_df)
    
    # Combine all data
    combined_df = pd.concat(all_token_data, ignore_index=True)
    
    # Analyze average generation length per model
    length_stats = combined_df.groupby('model')['generation_length'].agg(['mean', 'std', 'min', 'max']).reset_index()
    length_stats.columns = ['model', 'avg_generation_length', 'std_generation_length', 'min_generation_length', 'max_generation_length']
    
    # Save length statistics
    length_filename = os.path.join(output_dir, f'generation_length_stats_{timestamp}.csv')
    length_stats.to_csv(length_filename, index=False)
    print(f"Generation length statistics saved to: {length_filename}")
    
    # Analyze entropy changes for specific tokens
    token_entropy_changes = analyze_specific_token_changes(combined_df, output_dir, timestamp)
    
    # Analyze overall entropy statistics per model
    entropy_stats = combined_df.groupby('model')['entropy'].agg(['mean', 'std', 'min', 'max']).reset_index()
    entropy_stats.columns = ['model', 'avg_entropy', 'std_entropy', 'min_entropy', 'max_entropy']
    
    # Save entropy statistics
    entropy_filename = os.path.join(output_dir, f'entropy_stats_{timestamp}.csv')
    entropy_stats.to_csv(entropy_filename, index=False)
    print(f"Entropy statistics saved to: {entropy_filename}")
    
    # Analyze tokens with highest entropy variance across models
    token_variance_analysis = analyze_token_variance(combined_df, output_dir, timestamp)
    
    return {
        'length_stats': length_stats,
        'entropy_stats': entropy_stats,
        'token_entropy_changes': token_entropy_changes,
        'token_variance_analysis': token_variance_analysis
    }

def analyze_specific_token_changes(combined_df, output_dir, timestamp):
    """Analyze how specific tokens' entropy changed across models."""
    print(f"\nAnalyzing specific token entropy changes...")
    
    # Focus on common tokens that appear across multiple models
    token_counts = combined_df['current_token'].value_counts()
    common_tokens = token_counts[token_counts >= 10].index.tolist()  # Tokens that appear at least 10 times
    
    token_entropy_data = []
    
    for token in common_tokens[:100]:  # Limit to top 100 tokens for analysis
        token_data = combined_df[combined_df['current_token'] == token]
        
        if len(token_data) > 0:
            # Calculate entropy statistics for this token across models
            token_stats = token_data.groupby('model')['entropy'].agg(['mean', 'std', 'count']).reset_index()
            token_stats['token'] = token
            
            # Calculate entropy change (if we have multiple models)
            if len(token_stats) > 1:
                min_entropy = token_stats['mean'].min()
                max_entropy = token_stats['mean'].max()
                entropy_range = max_entropy - min_entropy
                token_stats['entropy_range'] = entropy_range
                token_stats['entropy_variance'] = token_stats['mean'].var()
            else:
                token_stats['entropy_range'] = 0
                token_stats['entropy_variance'] = 0
            
            token_entropy_data.append(token_stats)
    
    if token_entropy_data:
        token_entropy_df = pd.concat(token_entropy_data, ignore_index=True)
        
        # Save token entropy changes
        token_entropy_filename = os.path.join(output_dir, f'token_entropy_changes_{timestamp}.csv')
        token_entropy_df.to_csv(token_entropy_filename, index=False)
        print(f"Token entropy changes saved to: {token_entropy_filename}")
        
        # Find tokens with highest entropy variance
        token_variance = token_entropy_df.groupby('token')['entropy_variance'].first().sort_values(ascending=False)
        high_variance_tokens = token_variance.head(50)
        
        # Save high variance tokens
        high_variance_filename = os.path.join(output_dir, f'high_variance_tokens_{timestamp}.csv')
        high_variance_tokens.to_csv(high_variance_filename)
        print(f"High variance tokens saved to: {high_variance_filename}")
        
        return token_entropy_df
    
    return pd.DataFrame()

def analyze_token_variance(combined_df, output_dir, timestamp):
    """Analyze which tokens have the highest variance in entropy across models."""
    print(f"\nAnalyzing token variance across models...")
    
    # Group by token and calculate variance across models
    token_variance = combined_df.groupby('current_token')['entropy'].agg(['mean', 'std', 'var', 'count']).reset_index()
    token_variance.columns = ['token', 'mean_entropy', 'std_entropy', 'variance_entropy', 'occurrence_count']
    
    # Filter tokens that appear multiple times
    token_variance = token_variance[token_variance['occurrence_count'] >= 5]
    
    # Sort by variance
    token_variance = token_variance.sort_values('variance_entropy', ascending=False)
    
    # Save token variance analysis
    variance_filename = os.path.join(output_dir, f'token_variance_analysis_{timestamp}.csv')
    token_variance.to_csv(variance_filename, index=False)
    print(f"Token variance analysis saved to: {variance_filename}")
    
    # Save top tokens with highest variance
    top_variance_tokens = token_variance.head(100)
    top_variance_filename = os.path.join(output_dir, f'top_variance_tokens_{timestamp}.csv')
    top_variance_tokens.to_csv(top_variance_filename, index=False)
    print(f"Top variance tokens saved to: {top_variance_filename}")
    
    return token_variance

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
        generation_length = result['generation_length']
        
        # Add prompt information to each row
        for idx, row in stats_df.iterrows():
            stats_data = {
                'prompt_id': prompt_id,
                'full_text': full_text,
                'generation_length': generation_length,
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
    print(f"Average generation length: {final_df['generation_length'].mean():.2f} tokens")
    
    print(f"\nEntropy statistics:")
    print(final_df['entropy'].describe())
    
    print(f"\nProbability of chosen token statistics:")
    print(final_df['probability_of_next_token'].describe())
    
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
        'avg_generation_length': final_df['generation_length'].mean(),
        'std_generation_length': final_df['generation_length'].std(),
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
    
    # Clear model from memory
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'model_name': model_name,
        'model_name_clean': model_name_clean,
        'stats_file': output_filename,
        'summary_file': summary_filename,
        'total_tokens': len(final_df),
        'avg_generation_length': final_df['generation_length'].mean(),
        'mean_entropy': final_df['entropy'].mean(),
        'mean_probability': final_df['probability_of_next_token'].mean()
    }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze multiple language models for token-level statistics and entropy changes')
    parser.add_argument('--models', nargs='+', required=True, 
                       help='List of model paths or HuggingFace model names (space-separated)')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--output-dir', type=str, 
                       default="/tmpdir/m24047krsh/changes_in_reasoning/results",
                       help='Directory to save results')
    parser.add_argument('--max-prompts', type=int, default=1000,
                       help='Maximum number of prompts to process (default: 1000)')
    
    args = parser.parse_args()
    
    # Configuration
    MODEL_NAMES = args.models
    MAX_NEW_TOKENS = args.max_new_tokens
    MAX_PROMPTS = args.max_prompts
    DATA_FILE = "data_subset.jsonl"  # Use data_subset.jsonl for the first 1000 prompts
    OUTPUT_DIR = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load prompts from JSONL file (limited to first 1000)
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found!")
        return
    
    PROMPTS = load_prompts_from_jsonl(DATA_FILE, max_prompts=MAX_PROMPTS)
    
    if not PROMPTS:
        print("No prompts loaded! Exiting.")
        return
    
    print(f"Processing first {len(PROMPTS)} prompts from the dataset.")
    
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
    
    # Analyze token entropy changes across models
    if len(all_model_results) > 1:
        entropy_analysis = analyze_token_entropy_changes(all_model_results, OUTPUT_DIR, timestamp)
    
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
                'avg_generation_length': result['avg_generation_length'],
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
                  f"avg length: {result['avg_generation_length']:.2f}, "
                  f"mean entropy: {result['mean_entropy']:.4f}, "
                  f"mean probability: {result['mean_probability']:.4f}")
    
    print(f"\nAnalysis complete! All data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
