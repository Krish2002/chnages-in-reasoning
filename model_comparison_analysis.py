import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
import argparse
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer from a checkpoint path or HuggingFace model name."""
    print(f"Using device: {device}\n")
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_stats_for_generation(model, tokenizer, generated_ids, prompt_length):
    """Extract statistics for each token in the generation (ONLY generated tokens, not prompt)."""
    device = model.device
    inputs = {"input_ids": generated_ids.to(device)}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
        logits = outputs.logits

    stats = []
    # Start from prompt_length - 1 to get the first generated token's prediction
    # This ensures we only calculate stats on generated tokens, not on the prompt
    for j in range(prompt_length - 1, generated_ids.shape[1] - 1):
        pos_logits = logits[0, j, :]
        entropy = torch.distributions.Categorical(logits=pos_logits).entropy().item()
        actual_next_token_id = generated_ids[0, j + 1]
        probabilities = F.softmax(pos_logits, dim=-1)
        prob_of_chosen_token = probabilities[actual_next_token_id].item()
        current_token_str = tokenizer.decode(generated_ids[0, j])
        next_token_str = tokenizer.decode(actual_next_token_id)
        
        # Add generation step index for debugging/analysis
        generation_step = j - (prompt_length - 1) + 1  # 1-indexed generation step
        
        stats.append({
            'current_token': current_token_str,
            'next_token': next_token_str,
            'entropy': entropy,
            'probability_of_next_token': prob_of_chosen_token,
            'generation_step': generation_step,  # Which generation step this token was
        })
        
    return pd.DataFrame(stats)

def process_generations_multi_models(models, model_names, tokenizer, prompts, max_new_tokens=256):
    """Process generations for multiple models and return statistics."""
    all_outputs = []

    for i, prompt in enumerate(prompts):
        print(f"\n\n{'=' * 35} PROMPT {i + 1}/{len(prompts)} {'=' * 35}")
        formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        prompt_length = inputs.input_ids.shape[1]
        print(f"PROMPT: \"{formatted_prompt[:500]}...\"\n")
        print("-" * 30)
        
        prompt_results = {
            "prompt": formatted_prompt,
            "prompt_index": i
        }
        
        # Generate with each model
        for model, model_name in zip(models, model_names):
            print(f"Generating with {model_name}...")
            device = model.device
            generated_ids = model.generate(
                inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Debug info: show prompt vs generation lengths
            total_length = generated_ids.shape[1]
            num_generated = total_length - prompt_length
            print(f"  Prompt length: {prompt_length}, Total length: {total_length}, Generated tokens: {num_generated}")
            
            stats = get_stats_for_generation(model, tokenizer, generated_ids, prompt_length)
            full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            prompt_results[f"{model_name}_stats"] = stats
            prompt_results[f"{model_name}_text"] = full_text
            print(f"  Calculated stats for {len(stats)} generated tokens")
            print("Done.")
        
        print("-" * 30)
        all_outputs.append(prompt_results)
    
    return all_outputs

def load_prompts_from_jsonl(jsonl_path, num_examples=1000):
    """Load prompts from JSONL file, extracting system and user messages."""
    prompts = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
                
            try:
                data = json.loads(line.strip())
                conversations = data.get('conversations', [])
                
                # Extract system and user messages
                system_msg = None
                user_msg = None
                
                for msg in conversations:
                    if msg.get('role') == 'system':
                        system_msg = msg.get('content', '')
                    elif msg.get('role') == 'user':
                        user_msg = msg.get('content', '')
                        break  # Take the first user message
                
                if system_msg and user_msg:
                    prompt = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ]
                    prompts.append(prompt)
                    
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {i+1}")
                continue
    
    print(f"Loaded {len(prompts)} prompts from {jsonl_path}")
    return prompts

def analyze_token_qcuts(df, metric_column, model_name, n_qcuts=10):
    """Analyze token quantiles for a given metric."""
    print(f"\n{'=' * 30}")
    print(f"Token qcuts for {model_name} based on '{metric_column}'")
    print(f"{'=' * 30}")
    
    # Show sample of generated tokens to verify we're only analyzing generations
    print(f"Sample of generated tokens from {model_name}:")
    sample_tokens = df[['current_token', 'next_token', 'generation_step', metric_column]].head(10)
    print(sample_tokens.to_string(index=False))
    print(f"Generation steps range: {df['generation_step'].min()} to {df['generation_step'].max()}")
    
    qcut_col_name = f'{metric_column}_qcut'
    df[qcut_col_name] = pd.qcut(df[metric_column], n_qcuts, labels=False, duplicates='drop')
    qcut_intervals = pd.qcut(df[metric_column], n_qcuts, duplicates='drop')
    interval_map = {i: interval for i, interval in enumerate(qcut_intervals.cat.categories)}
    
    num_actual_qcuts = df[qcut_col_name].nunique()
    qcuts_to_display = sorted(list(set([0, num_actual_qcuts // 4, num_actual_qcuts // 2, 
                                       num_actual_qcuts * 3 // 4, num_actual_qcuts - 1])))

    for qcut_idx in qcuts_to_display:
        df_qcut = df[df[qcut_col_name] == qcut_idx].copy()
        df_qcut.sort_values(by=metric_column, inplace=True)
        print(f"\n--- qcut {qcut_idx + 1}/{num_actual_qcuts} (range: {interval_map[qcut_idx]}) ---")
        num_samples = min(5, len(df_qcut))
        step = max(1, len(df_qcut) // num_samples)
        
        print(df_qcut[['current_token', 'next_token', 'generation_step', metric_column]].iloc[::step])

def plot_entropy_comparison_multi_models(all_results, model_names, output_dir="plots"):
    """Create plots comparing entropy distributions between multiple models."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Aggregate stats across all prompts for each model
    aggregated_stats = {}
    for model_name in model_names:
        all_stats = [result[f'{model_name}_stats'] for result in all_results]
        aggregated_stats[model_name] = pd.concat(all_stats, ignore_index=True)
        print(f"\nAggregated a total of {len(aggregated_stats[model_name])} tokens for {model_name}.")
    
    # Plot aggregated entropy distribution for all models
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, model_name in enumerate(model_names):
        color = colors[i % len(colors)]
        aggregated_stats[model_name]['entropy'].hist(
            ax=ax, bins=50, alpha=0.6, 
            label=f'{model_name} - all prompts', 
            density=True, log=True, color=color
        )
    
    ax.set_title('Aggregated distribution of token entropies across all generations')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Density')
    ax.legend()
    plt.savefig(f"{output_dir}/aggregated_entropy_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create entropy over generation steps plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Sample a few prompts for visualization
    sample_prompts = min(3, len(all_results))
    for i in range(sample_prompts):
        result = all_results[i]
        for j, model_name in enumerate(model_names):
            if j < len(axes):
                axes[j].plot(
                    result[f'{model_name}_stats']['entropy'].values, 
                    alpha=0.7, 
                    label=f'Prompt {i+1}',
                    color=colors[i % len(colors)]
                )
                axes[j].set_title(f'{model_name}: Entropy over generation steps')
                axes[j].set_ylabel('Entropy')
                axes[j].legend()
                axes[j].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(len(model_names), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/entropy_over_steps.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    for model_name in model_names:
        print(f"\n{model_name} entropy statistics:")
        print(aggregated_stats[model_name]['entropy'].describe())
    
    return aggregated_stats

def main():
    parser = argparse.ArgumentParser(description='Compare multiple model generations and analyze entropy statistics')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, 
                       help='Paths to model checkpoints (space-separated)')
    parser.add_argument('--model_names', type=str, nargs='+', required=True,
                       help='Names for the models (space-separated, same order as paths)')
    parser.add_argument('--data_path', type=str, default='data_full.jsonl', help='Path to JSONL data file')
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of examples to process')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum new tokens to generate')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--save_detailed_tokens', action='store_true', 
                       help='Save detailed token-level stats for each prompt (can be large)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.model_paths) != len(args.model_names):
        print("Error: Number of model paths must match number of model names!")
        return
    
    if len(args.model_paths) < 2:
        print("Error: At least 2 models are required for comparison!")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load models
    print("Loading models...")
    models = []
    for model_path in args.model_paths:
        model, tokenizer = load_model_and_tokenizer(model_path)
        models.append(model)
    
    # Load prompts from JSONL file
    prompts = load_prompts_from_jsonl(args.data_path, args.num_examples)
    
    # Process generations
    all_results = process_generations_multi_models(models, args.model_names, tokenizer, prompts, args.max_new_tokens)
    
    # Analyze and plot results
    aggregated_stats = plot_entropy_comparison_multi_models(all_results, args.model_names, args.output_dir)
    
    # Analyze token quantiles for each model
    for model_name in args.model_names:
        analyze_token_qcuts(aggregated_stats[model_name], 'entropy', f'{model_name} - aggregate', n_qcuts=10)
    
    # Save detailed results
    print("Saving detailed results...")
    
    # Save aggregated token statistics for each model
    for model_name in args.model_names:
        aggregated_stats[model_name].to_csv(f"{args.output_dir}/{model_name}_token_stats.csv", index=False)
    
    # Save prompt-level results
    prompt_results = []
    for i, result in enumerate(all_results):
        prompt_result = {
            'prompt_index': i,
            'prompt': result['prompt'][:1000],  # Truncate for readability
        }
        
        # Add stats for each model
        for model_name in args.model_names:
            prompt_result[f'{model_name}_text'] = result[f'{model_name}_text']
            prompt_result[f'{model_name}_num_tokens'] = len(result[f'{model_name}_stats'])
            prompt_result[f'{model_name}_mean_entropy'] = result[f'{model_name}_stats']['entropy'].mean()
            prompt_result[f'{model_name}_std_entropy'] = result[f'{model_name}_stats']['entropy'].std()
            prompt_result[f'{model_name}_median_entropy'] = result[f'{model_name}_stats']['entropy'].median()
        
        prompt_results.append(prompt_result)
    
    # Save prompt-level summary
    prompt_df = pd.DataFrame(prompt_results)
    prompt_df.to_csv(f"{args.output_dir}/prompt_level_results.csv", index=False)
    
    # Save detailed token stats for each model (all prompts combined)
    if args.save_detailed_tokens:
        print("Saving detailed token-level statistics...")
        for model_name in args.model_names:
            # Combine all token stats for this model with prompt index
            all_tokens_for_model = []
            for i, result in enumerate(all_results):
                token_stats = result[f'{model_name}_stats'].copy()
                token_stats['prompt_index'] = i
                all_tokens_for_model.append(token_stats)
            
            combined_tokens = pd.concat(all_tokens_for_model, ignore_index=True)
            combined_tokens.to_csv(f"{args.output_dir}/{model_name}_all_tokens_detailed.csv", index=False)
    
    # Save summary statistics
    results_summary = {
        'model_paths': dict(zip(args.model_names, args.model_paths)),
        'num_examples_processed': len(prompts),
        'max_new_tokens': args.max_new_tokens,
    }
    
    # Add statistics for each model
    for model_name in args.model_names:
        stats = aggregated_stats[model_name]
        results_summary[f'{model_name}_total_tokens'] = len(stats)
        results_summary[f'{model_name}_mean_entropy'] = stats['entropy'].mean()
        results_summary[f'{model_name}_std_entropy'] = stats['entropy'].std()
        results_summary[f'{model_name}_median_entropy'] = stats['entropy'].median()
        results_summary[f'{model_name}_min_entropy'] = stats['entropy'].min()
        results_summary[f'{model_name}_max_entropy'] = stats['entropy'].max()
    
    with open(f"{args.output_dir}/results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/")
    print("Summary:")
    for key, value in results_summary.items():
        if key != 'model_paths':  # Don't print the full paths in summary
            print(f"  {key}: {value}")
    
    print(f"\nModel paths saved in {args.output_dir}/results_summary.json")

if __name__ == "__main__":
    main() 