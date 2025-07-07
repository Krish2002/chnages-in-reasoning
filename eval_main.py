#!/usr/bin/env python3
"""
Evaluation script for comparing checkpoint models.
Analyzes token probability changes across training checkpoints.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import argparse
from typing import List, Dict, Any
from collections import defaultdict

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer from path or HuggingFace repo."""
    print(f"Using device: {device}\n")
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_prompts_from_jsonl(file_path: str, max_prompts: int = 1000):
    """Load prompts from a JSONL file."""
    print(f"Loading prompts from {file_path}")
    prompts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_prompts:
                    break
                try:
                    data = json.loads(line.strip())
                    # Handle different possible formats
                    if isinstance(data, dict):
                        if 'conversations' in data:
                            # Format: {"conversations": [{"role": "...", "content": "..."}, ...]}
                            conversations = data['conversations']
                            if isinstance(conversations, list) and len(conversations) > 0:
                                prompts.append(conversations)
                        elif 'messages' in data:
                            prompts.append(data['messages'])
                        elif 'prompt' in data:
                            # If it's a single prompt, wrap it in messages format
                            prompts.append([
                                {"role": "system", "content": "You are a helpful assistant. Think step by step before responding to the user's query. Your thought process should be enclosed between <think> and </think> tags. Once your thought process is complete, write a response which should end in the final answer enclosed in \\boxed{}."},
                                {"role": "user", "content": data['prompt']}
                            ])
                        elif 'user' in data or 'assistant' in data:
                            # If it's already in messages format
                            prompts.append([data])
                        else:
                            # Try to find any text content
                            for key, value in data.items():
                                if isinstance(value, str) and len(value) > 10:
                                    prompts.append([
                                        {"role": "system", "content": "You are a helpful assistant. Think step by step before responding to the user's query. Your thought process should be enclosed between <think> and </think> tags. Once your thought process is complete, write a response which should end in the final answer enclosed in \\boxed{}."},
                                        {"role": "user", "content": value}
                                    ])
                                    break
                    elif isinstance(data, list):
                        # If it's already a list of messages
                        prompts.append(data)
                    else:
                        print(f"Warning: Skipping line {i+1} - unexpected format")
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {i+1} - JSON decode error: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    print(f"Successfully loaded {len(prompts)} prompts from {file_path}")
    return prompts

def get_stats_for_generation(model, tokenizer, generated_ids, prompt_length):
    """Extract statistics for each token in the generation."""
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

def process_generations(models, model_names, tokenizer, prompts: List[List[Dict[str, str]]], max_new_tokens: int = 256):
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
        }
        
        # Process each model
        for model_idx, (model, model_name) in enumerate(zip(models, model_names)):
            print(f"Generating with {model_name}...")
            device = model.device
            generated_ids = model.generate(
                inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
            stats = get_stats_for_generation(model, tokenizer, generated_ids, prompt_length)
            full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print("Done.\n")
            
            prompt_results[f"{model_name}_stats"] = stats
            prompt_results[f"{model_name}_text"] = full_text
        
        print("-" * 30)
        all_outputs.append(prompt_results)
    
    return all_outputs

def analyze_checkpoint_probability_changes(all_results, model_names, output_dir: str, experiment_name: str):
    """Analyze probability changes across checkpoints."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*50)
    print("CHECKPOINT PROBABILITY ANALYSIS")
    print("="*50)
    
    # 1. Average output token probability change for each checkpoint
    print("\n1. AVERAGE OUTPUT TOKEN PROBABILITY FOR EACH CHECKPOINT:")
    print("-" * 60)
    
    checkpoint_avg_probs = {}
    for model_name in model_names:
        all_stats = [result[f'{model_name}_stats'] for result in all_results]
        agg_df = pd.concat(all_stats, ignore_index=True)
        avg_prob = agg_df['probability_of_next_token'].mean()
        checkpoint_avg_probs[model_name] = avg_prob
        print(f"{model_name}: {avg_prob:.6f}")
    
    # 2. Token probability changes between checkpoints
    print("\n2. TOKEN PROBABILITY CHANGES BETWEEN CHECKPOINTS:")
    print("-" * 60)
    
    # Aggregate all token probabilities by model
    token_probs_by_model = {}
    for model_name in model_names:
        all_stats = [result[f'{model_name}_stats'] for result in all_results]
        agg_df = pd.concat(all_stats, ignore_index=True)
        # Group by next_token and calculate mean probability
        token_probs = agg_df.groupby('next_token')['probability_of_next_token'].mean().to_dict()
        token_probs_by_model[model_name] = token_probs
    
    # Calculate changes between consecutive checkpoints
    checkpoint_changes = {}
    for i in range(len(model_names) - 1):
        checkpoint1 = model_names[i]
        checkpoint2 = model_names[i + 1]
        
        print(f"\nChanges from {checkpoint1} to {checkpoint2}:")
        print(f"{'Token':<20} {'Old Prob':<12} {'New Prob':<12} {'Change':<12}")
        print("-" * 60)
        
        changes = []
        tokens1 = set(token_probs_by_model[checkpoint1].keys())
        tokens2 = set(token_probs_by_model[checkpoint2].keys())
        all_tokens = tokens1.union(tokens2)
        
        for token in sorted(all_tokens):
            prob1 = token_probs_by_model[checkpoint1].get(token, 0.0)
            prob2 = token_probs_by_model[checkpoint2].get(token, 0.0)
            change = prob2 - prob1
            
            if abs(change) > 0.01:  # Only show significant changes
                changes.append({
                    'token': token,
                    'old_prob': prob1,
                    'new_prob': prob2,
                    'change': change
                })
                print(f"{token:<20} {prob1:<12.6f} {prob2:<12.6f} {change:<+12.6f}")
        
        checkpoint_changes[f"{checkpoint1}_to_{checkpoint2}"] = changes
    
    # 3. Histogram data for each model
    print("\n3. TOKEN PROBABILITY HISTOGRAM DATA FOR EACH MODEL:")
    print("-" * 60)
    
    histogram_data = {}
    for model_name in model_names:
        all_stats = [result[f'{model_name}_stats'] for result in all_results]
        agg_df = pd.concat(all_stats, ignore_index=True)
        
        # Create histogram bins
        prob_values = agg_df['probability_of_next_token'].values
        hist, bin_edges = np.histogram(prob_values, bins=50, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        histogram_data[model_name] = {
            'probabilities': prob_values.tolist(),
            'histogram_counts': hist.tolist(),
            'bin_centers': bin_centers.tolist(),
            'bin_edges': bin_edges.tolist(),
            'mean_probability': float(np.mean(prob_values)),
            'std_probability': float(np.std(prob_values)),
            'min_probability': float(np.min(prob_values)),
            'max_probability': float(np.max(prob_values)),
            'median_probability': float(np.median(prob_values)),
            'total_tokens': len(prob_values)
        }
        
        print(f"\n{model_name}:")
        print(f"  Mean probability: {histogram_data[model_name]['mean_probability']:.6f}")
        print(f"  Std probability: {histogram_data[model_name]['std_probability']:.6f}")
        print(f"  Min probability: {histogram_data[model_name]['min_probability']:.6f}")
        print(f"  Max probability: {histogram_data[model_name]['max_probability']:.6f}")
        print(f"  Median probability: {histogram_data[model_name]['median_probability']:.6f}")
        print(f"  Total tokens: {histogram_data[model_name]['total_tokens']}")
    
    # Save all analysis results
    analysis_results = {
        'timestamp': timestamp,
        'experiment_name': experiment_name,
        'checkpoint_average_probabilities': checkpoint_avg_probs,
        'checkpoint_probability_changes': checkpoint_changes,
        'histogram_data': histogram_data,
        'token_probabilities_by_model': token_probs_by_model
    }
    
    # Save detailed results
    with open(os.path.join(output_dir, f"checkpoint_analysis_{timestamp}.json"), 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Save CSV files for easy analysis
    # 1. Average probabilities by checkpoint
    avg_prob_df = pd.DataFrame(list(checkpoint_avg_probs.items()), 
                              columns=['checkpoint', 'average_probability'])
    avg_prob_df.to_csv(os.path.join(output_dir, f"checkpoint_average_probabilities_{timestamp}.csv"), index=False)
    
    # 2. Token probability changes
    all_changes = []
    for change_key, changes in checkpoint_changes.items():
        for change in changes:
            change['checkpoint_pair'] = change_key
            all_changes.append(change)
    
    if all_changes:
        changes_df = pd.DataFrame(all_changes)
        changes_df.to_csv(os.path.join(output_dir, f"token_probability_changes_{timestamp}.csv"), index=False)
    
    # 3. Histogram data
    for model_name, hist_data in histogram_data.items():
        hist_df = pd.DataFrame({
            'bin_center': hist_data['bin_centers'],
            'count': hist_data['histogram_counts']
        })
        hist_df.to_csv(os.path.join(output_dir, f"{model_name}_probability_histogram_{timestamp}.csv"), index=False)
    
    print(f"\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Results saved to {output_dir}")
    print(f"Timestamp: {timestamp}")
    
    return analysis_results

def save_results(all_results, model_names, output_dir: str, experiment_name: str):
    """Save all results to files for later analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual prompt results
    for i, result in enumerate(all_results):
        prompt_dir = os.path.join(output_dir, f"prompt_{i+1}")
        os.makedirs(prompt_dir, exist_ok=True)
        
        # Save stats for each model
        for model_name in model_names:
            result[f'{model_name}_stats'].to_csv(
                os.path.join(prompt_dir, f"{model_name}_stats_{timestamp}.csv"), 
                index=False
            )
        
        # Save generated texts
        texts_data = {'prompt': result['prompt']}
        for model_name in model_names:
            texts_data[f'{model_name}_text'] = result[f'{model_name}_text']
        
        with open(os.path.join(prompt_dir, f"generated_texts_{timestamp}.json"), 'w') as f:
            json.dump(texts_data, f, indent=2)
    
    # Aggregate statistics for each model
    aggregated_stats = {}
    summary_stats = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'total_prompts': len(all_results),
        'models': model_names,
    }
    
    for model_name in model_names:
        all_stats = [result[f'{model_name}_stats'] for result in all_results]
        agg_df = pd.concat(all_stats, ignore_index=True)
        aggregated_stats[model_name] = agg_df
        
        # Save aggregated data
        agg_df.to_csv(os.path.join(output_dir, f"aggregated_{model_name}_stats_{timestamp}.csv"), index=False)
        
        # Add to summary statistics
        summary_stats[f'{model_name}_total_tokens'] = len(agg_df)
        summary_stats[f'{model_name}_entropy_stats'] = agg_df['entropy'].describe().to_dict()
        summary_stats[f'{model_name}_probability_stats'] = agg_df['probability_of_next_token'].describe().to_dict()
    
    # Save summary statistics
    with open(os.path.join(output_dir, f"summary_stats_{timestamp}.json"), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Run checkpoint-specific analysis
    checkpoint_analysis = analyze_checkpoint_probability_changes(all_results, model_names, output_dir, experiment_name)
    
    print(f"\nResults saved to {output_dir}")
    print(f"Timestamp: {timestamp}")
    for model_name in model_names:
        print(f"Total tokens processed - {model_name}: {len(aggregated_stats[model_name])}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint models - token probability analysis')
    parser.add_argument('--model_1', type=str, required=True, help='Path to Checkpoint 1 (earliest)')
    parser.add_argument('--model_2', type=str, required=True, help='Path to Checkpoint 2')
    parser.add_argument('--model_3', type=str, required=True, help='Path to Checkpoint 3')
    parser.add_argument('--model_4', type=str, required=True, help='Path to Checkpoint 4 (latest)')
    parser.add_argument('--checkpoint_names', type=str, nargs=4, default=None, 
                       help='Names for the checkpoints (default: checkpoint_1, checkpoint_2, checkpoint_3, checkpoint_4)')
    parser.add_argument('--data_file', type=str, default='/tmpdir/m24047krsh/changes_in_reasoning/data_full.jsonl', 
                       help='Path to JSONL file containing prompts (default: data_full.jsonl)')
    parser.add_argument('--max_prompts', type=int, default=1000, 
                       help='Maximum number of prompts to process (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='/tmpdir/m24047krsh/changes_in_reasoning/results', help='Output directory for results')
    parser.add_argument('--max_new_tokens', type=int, default=4096, help='Maximum new tokens to generate')
    parser.add_argument('--experiment_name', type=str, default='checkpoint_analysis', help='Name for this experiment')
    
    args = parser.parse_args()
    
    # Set checkpoint names
    if args.checkpoint_names:
        checkpoint_names = args.checkpoint_names
    else:
        checkpoint_names = ['checkpoint_1', 'checkpoint_2', 'checkpoint_3', 'checkpoint_4']
    
    # Load models
    models = []
    for i, model_path in enumerate([args.model_1, args.model_2, args.model_3, args.model_4]):
        print(f"\nLoading {checkpoint_names[i]} from {model_path}")
        model, tokenizer = load_model_and_tokenizer(model_path)
        models.append(model)
        if i == 0:  # Use tokenizer from first model for all
            base_tokenizer = tokenizer
    
    # Load prompts from JSONL file
    print(f"\nLoading prompts from {args.data_file}")
    PROMPTS = load_prompts_from_jsonl(args.data_file, max_prompts=args.max_prompts)
    
    if not PROMPTS:
        print("Error: No prompts loaded. Please check the data file format.")
        return
    
    print(f"Processing {len(PROMPTS)} prompts from {args.data_file}")
    
    # Run generation experiments
    print("Running generation evaluation...")
    all_results = process_generations(models, checkpoint_names, base_tokenizer, PROMPTS, max_new_tokens=args.max_new_tokens)
    
    # Save results and run checkpoint analysis
    save_results(all_results, checkpoint_names, args.output_dir, args.experiment_name)
    
    print("\nCheckpoint analysis completed successfully!")

if __name__ == "__main__":
    main() 