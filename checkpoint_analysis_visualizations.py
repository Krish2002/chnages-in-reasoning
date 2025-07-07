#!/usr/bin/env python3
"""
Checkpoint Analysis Visualizations
Analyzes token probability changes across training checkpoints.
Focuses on average length, entropy changes, and most changed tokens.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import glob

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_checkpoint_data(results_dir="results"):
    """Load all checkpoint data from the results directory."""
    print("Loading checkpoint data...")
    
    # Find the most recent timestamp from the files
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    
    # Extract timestamps from filenames
    timestamps = []
    for file in csv_files:
        if "_stats_" in file:
            parts = file.split("_stats_")
            if len(parts) > 1:
                timestamp_str = parts[1].replace(".csv", "")
                timestamps.append(timestamp_str)
    
    if not timestamps:
        raise FileNotFoundError("No model stats files found")
    
    latest_timestamp = max(timestamps)
    print(f"Using data from timestamp: {latest_timestamp}")
    
    # Load summary statistics
    summary_file = os.path.join(results_dir, f"summary_stats_{latest_timestamp}.json")
    with open(summary_file, 'r') as f:
        summary_stats = json.load(f)
    
    # Load average probabilities
    avg_prob_file = os.path.join(results_dir, f"checkpoint_average_probabilities_{latest_timestamp.replace('180134', '180144')}.csv")
    avg_probs_df = pd.read_csv(avg_prob_file)
    
    # Load token probability changes
    changes_file = os.path.join(results_dir, f"token_probability_changes_{latest_timestamp.replace('180134', '180144')}.csv")
    changes_df = pd.read_csv(changes_file)
    
    # Load histogram data for each checkpoint
    histogram_data = {}
    for checkpoint in ['checkpoint_1', 'checkpoint_2', 'checkpoint_3', 'checkpoint_4']:
        hist_file = os.path.join(results_dir, f"{checkpoint}_probability_histogram_{latest_timestamp.replace('180134', '180144')}.csv")
        if os.path.exists(hist_file):
            histogram_data[checkpoint] = pd.read_csv(hist_file)
    
    return summary_stats, avg_probs_df, changes_df, histogram_data, latest_timestamp

def plot_average_length_analysis(summary_stats):
    """Plot 1: Average generation length analysis across checkpoints."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    checkpoints = summary_stats['models']
    total_tokens = [summary_stats[f'{cp}_total_tokens'] for cp in checkpoints]
    avg_tokens_per_prompt = [tokens / summary_stats['total_prompts'] for tokens in total_tokens]
    
    # Plot 1: Total tokens generated per checkpoint
    ax1.bar(checkpoints, total_tokens, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    ax1.set_title('Total Tokens Generated per Checkpoint', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Tokens')
    ax1.set_xlabel('Checkpoint')
    for i, v in enumerate(total_tokens):
        ax1.text(i, v + max(total_tokens)*0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average tokens per prompt
    ax2.bar(checkpoints, avg_tokens_per_prompt, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    ax2.set_title('Average Tokens per Prompt', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Tokens')
    ax2.set_xlabel('Checkpoint')
    for i, v in enumerate(avg_tokens_per_prompt):
        ax2.text(i, v + max(avg_tokens_per_prompt)*0.01, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Length distribution comparison
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, cp in enumerate(checkpoints):
        # Create synthetic distribution for visualization
        synthetic_data = np.random.normal(avg_tokens_per_prompt[i], avg_tokens_per_prompt[i]*0.1, 1000)
        ax3.hist(synthetic_data, bins=20, alpha=0.6, color=colors[i], label=cp)
    ax3.set_title('Distribution of Average Lengths', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Average Tokens per Prompt')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Length trend
    ax4.plot(checkpoints, avg_tokens_per_prompt, 'o-', linewidth=3, markersize=8, color='#2ca02c')
    ax4.set_title('Length Trend Across Checkpoints', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Average Tokens per Prompt')
    ax4.set_xlabel('Checkpoint')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(range(len(checkpoints)), avg_tokens_per_prompt, 1)
    p = np.poly1d(z)
    ax4.plot(checkpoints, p(range(len(checkpoints))), '--', alpha=0.7, color='red')
    
    plt.tight_layout()
    plt.savefig('plots/average_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_entropy_changes_analysis(summary_stats):
    """Plot 2: Entropy changes analysis across checkpoints."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract entropy data
    checkpoints = summary_stats['models']
    entropy_means = [summary_stats[f'{cp}_entropy_stats']['mean'] for cp in checkpoints]
    entropy_stds = [summary_stats[f'{cp}_entropy_stats']['std'] for cp in checkpoints]
    entropy_medians = [summary_stats[f'{cp}_entropy_stats']['50%'] for cp in checkpoints]
    
    # Plot 1: Mean entropy across checkpoints
    ax1.bar(checkpoints, entropy_means, yerr=entropy_stds, capsize=5, 
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    ax1.set_title('Mean Entropy Across Checkpoints', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Entropy')
    ax1.set_xlabel('Checkpoint')
    for i, v in enumerate(entropy_means):
        ax1.text(i, v + entropy_stds[i] + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Entropy distribution comparison
    for i, cp in enumerate(checkpoints):
        # Create synthetic data based on mean and std for visualization
        synthetic_data = np.random.normal(entropy_means[i], entropy_stds[i], 1000)
        ax2.hist(synthetic_data, bins=30, alpha=0.5, label=cp, density=True)
    
    ax2.set_title('Entropy Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Entropy')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Entropy trend with confidence intervals
    ax3.errorbar(checkpoints, entropy_means, yerr=entropy_stds, fmt='o-', 
                linewidth=3, markersize=8, capsize=5, capthick=2)
    ax3.set_title('Entropy Trend with Standard Deviation', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mean Entropy')
    ax3.set_xlabel('Checkpoint')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Median vs Mean entropy
    x = np.arange(len(checkpoints))
    width = 0.35
    
    ax4.bar(x - width/2, entropy_means, width, label='Mean', alpha=0.7)
    ax4.bar(x + width/2, entropy_medians, width, label='Median', alpha=0.7)
    ax4.set_title('Mean vs Median Entropy', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Entropy')
    ax4.set_xlabel('Checkpoint')
    ax4.set_xticks(x)
    ax4.set_xticklabels(checkpoints)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/entropy_changes_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_token_probability_changes(changes_df):
    """Plot 3: Analysis of token probability changes."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter for significant changes (absolute change > 0.1)
    significant_changes = changes_df[abs(changes_df['change']) > 0.1].copy()
    
    # Plot 1: Distribution of probability changes
    ax1.hist(changes_df['change'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Token Probability Changes', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Probability Change')
    ax1.set_ylabel('Number of Tokens')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Change')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top tokens with largest increases
    top_increases = changes_df.nlargest(20, 'change')
    ax2.barh(range(len(top_increases)), top_increases['change'], color='green', alpha=0.7)
    ax2.set_yticks(range(len(top_increases)))
    ax2.set_yticklabels([f"'{token}'" for token in top_increases['token']], fontsize=8)
    ax2.set_title('Top 20 Tokens with Largest Probability Increases', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Probability Change')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Top tokens with largest decreases
    top_decreases = changes_df.nsmallest(20, 'change')
    ax3.barh(range(len(top_decreases)), top_decreases['change'], color='red', alpha=0.7)
    ax3.set_yticks(range(len(top_decreases)))
    ax3.set_yticklabels([f"'{token}'" for token in top_decreases['token']], fontsize=8)
    ax3.set_title('Top 20 Tokens with Largest Probability Decreases', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Probability Change')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Changes by checkpoint pair
    checkpoint_pairs = changes_df['checkpoint_pair'].unique()
    pair_stats = []
    for pair in checkpoint_pairs:
        pair_data = changes_df[changes_df['checkpoint_pair'] == pair]
        pair_stats.append({
            'pair': pair,
            'mean_change': pair_data['change'].mean(),
            'std_change': pair_data['change'].std(),
            'num_tokens': len(pair_data)
        })
    
    pair_df = pd.DataFrame(pair_stats)
    ax4.bar(pair_df['pair'], pair_df['mean_change'], yerr=pair_df['std_change'], 
            capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax4.set_title('Average Probability Changes by Checkpoint Pair', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Mean Probability Change')
    ax4.set_xlabel('Checkpoint Pair')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/token_probability_changes_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_most_changed_tokens(changes_df):
    """Plot 4: Detailed analysis of the most changed tokens."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get the most changed tokens (top 30 by absolute change)
    most_changed = pd.concat([
        changes_df.nlargest(30, 'change'),
        changes_df.nsmallest(30, 'change')
    ]).drop_duplicates()
    
    # Plot 1: Scatter plot of old vs new probabilities
    colors = ['red' if change < 0 else 'green' for change in most_changed['change']]
    ax1.scatter(most_changed['old_prob'], most_changed['new_prob'], 
               c=colors, alpha=0.6, s=50)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No Change')
    ax1.set_title('Old vs New Probabilities for Most Changed Tokens', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Old Probability')
    ax1.set_ylabel('New Probability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Token length vs probability change
    token_lengths = [len(str(token)) for token in most_changed['token']]
    ax2.scatter(token_lengths, most_changed['change'], alpha=0.6, s=50)
    ax2.set_title('Token Length vs Probability Change', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Token Length')
    ax2.set_ylabel('Probability Change')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of changes for different token types
    # Categorize tokens
    def categorize_token(token):
        token_str = str(token)
        if token_str.isalpha():
            return 'Alphabetic'
        elif token_str.isdigit():
            return 'Numeric'
        elif token_str.isspace():
            return 'Whitespace'
        elif any(c in token_str for c in '.,;:!?()[]{}"\''):
            return 'Punctuation'
        else:
            return 'Mixed'
    
    most_changed['token_type'] = most_changed['token'].apply(categorize_token)
    
    # Box plot by token type
    token_types = most_changed['token_type'].unique()
    type_data = [most_changed[most_changed['token_type'] == t]['change'] for t in token_types]
    ax3.boxplot(type_data, labels=token_types)
    ax3.set_title('Probability Changes by Token Type', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Probability Change')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Top 15 most changed tokens with their values
    top_15 = most_changed.nlargest(15, 'change')
    ax4.barh(range(len(top_15)), top_15['change'], color='green', alpha=0.7)
    ax4.set_yticks(range(len(top_15)))
    ax4.set_yticklabels([f"'{token}' ({old:.3f}→{new:.3f})" 
                        for token, old, new in zip(top_15['token'], top_15['old_prob'], top_15['new_prob'])], 
                       fontsize=8)
    ax4.set_title('Top 15 Most Changed Tokens with Values', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Probability Change')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/most_changed_tokens_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_probability_histograms(histogram_data):
    """Plot 5: Probability histogram comparison across checkpoints."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    checkpoints = list(histogram_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot 1: Overlaid histograms
    for i, (checkpoint, hist_df) in enumerate(histogram_data.items()):
        ax1.plot(hist_df['bin_center'], hist_df['count'], 
                label=checkpoint, color=colors[i], linewidth=2, alpha=0.8)
    ax1.set_title('Probability Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Probability')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Normalized histograms
    for i, (checkpoint, hist_df) in enumerate(histogram_data.items()):
        normalized_counts = hist_df['count'] / hist_df['count'].sum()
        ax2.plot(hist_df['bin_center'], normalized_counts, 
                label=checkpoint, color=colors[i], linewidth=2, alpha=0.8)
    ax2.set_title('Normalized Probability Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Normalized Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative distribution
    for i, (checkpoint, hist_df) in enumerate(histogram_data.items()):
        cumulative = np.cumsum(hist_df['count']) / hist_df['count'].sum()
        ax3.plot(hist_df['bin_center'], cumulative, 
                label=checkpoint, color=colors[i], linewidth=2, alpha=0.8)
    ax3.set_title('Cumulative Probability Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Probability')
    ax3.set_ylabel('Cumulative Probability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution statistics comparison
    stats_data = []
    for checkpoint, hist_df in histogram_data.items():
        # Calculate statistics from histogram
        total_count = hist_df['count'].sum()
        mean_prob = np.average(hist_df['bin_center'], weights=hist_df['count'])
        variance = np.average((hist_df['bin_center'] - mean_prob)**2, weights=hist_df['count'])
        std_prob = np.sqrt(variance)
        
        stats_data.append({
            'checkpoint': checkpoint,
            'mean': mean_prob,
            'std': std_prob,
            'total_count': total_count
        })
    
    stats_df = pd.DataFrame(stats_data)
    ax4.bar(stats_df['checkpoint'], stats_df['mean'], yerr=stats_df['std'], 
            capsize=5, alpha=0.7, color=colors[:len(stats_df)])
    ax4.set_title('Mean Probability with Standard Deviation', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Mean Probability')
    ax4.set_xlabel('Checkpoint')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/probability_histograms_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(summary_stats, avg_probs_df, changes_df, timestamp):
    """Create a comprehensive summary report."""
    print(f"\n{'='*80}")
    print(f"CHECKPOINT ANALYSIS SUMMARY REPORT - {timestamp}")
    print(f"{'='*80}")
    
    print(f"\nDataset Overview:")
    print(f"  Total prompts processed: {summary_stats['total_prompts']:,}")
    print(f"  Checkpoints analyzed: {len(summary_stats['models'])}")
    
    print(f"\nGeneration Length Analysis:")
    for cp in summary_stats['models']:
        total_tokens = summary_stats[f'{cp}_total_tokens']
        avg_length = total_tokens / summary_stats['total_prompts']
        print(f"  {cp}: {total_tokens:,} total tokens, {avg_length:.1f} avg tokens/prompt")
    
    print(f"\nEntropy Analysis:")
    for cp in summary_stats['models']:
        entropy_stats = summary_stats[f'{cp}_entropy_stats']
        print(f"  {cp}: mean={entropy_stats['mean']:.4f}, std={entropy_stats['std']:.4f}, median={entropy_stats['50%']:.4f}")
    
    print(f"\nProbability Analysis:")
    for cp in summary_stats['models']:
        prob_stats = summary_stats[f'{cp}_probability_stats']
        print(f"  {cp}: mean={prob_stats['mean']:.4f}, std={prob_stats['std']:.4f}, median={prob_stats['50%']:.4f}")
    
    print(f"\nToken Probability Changes:")
    print(f"  Total tokens with changes: {len(changes_df):,}")
    print(f"  Tokens with significant changes (>0.1): {len(changes_df[abs(changes_df['change']) > 0.1]):,}")
    print(f"  Average change magnitude: {abs(changes_df['change']).mean():.4f}")
    
    # Most changed tokens
    top_increases = changes_df.nlargest(5, 'change')
    top_decreases = changes_df.nsmallest(5, 'change')
    
    print(f"\nTop 5 Probability Increases:")
    for _, row in top_increases.iterrows():
        print(f"  '{row['token']}': {row['old_prob']:.3f} → {row['new_prob']:.3f} (+{row['change']:.3f})")
    
    print(f"\nTop 5 Probability Decreases:")
    for _, row in top_decreases.iterrows():
        print(f"  '{row['token']}': {row['old_prob']:.3f} → {row['new_prob']:.3f} ({row['change']:.3f})")
    
    # Save summary to file
    summary_file = f'plots/checkpoint_analysis_summary_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write(f"CHECKPOINT ANALYSIS SUMMARY REPORT - {timestamp}\n")
        f.write("="*80 + "\n")
        f.write(f"\nDataset Overview:\n")
        f.write(f"  Total prompts processed: {summary_stats['total_prompts']:,}\n")
        f.write(f"  Checkpoints analyzed: {len(summary_stats['models'])}\n")
        
        f.write(f"\nGeneration Length Analysis:\n")
        for cp in summary_stats['models']:
            total_tokens = summary_stats[f'{cp}_total_tokens']
            avg_length = total_tokens / summary_stats['total_prompts']
            f.write(f"  {cp}: {total_tokens:,} total tokens, {avg_length:.1f} avg tokens/prompt\n")
        
        f.write(f"\nEntropy Analysis:\n")
        for cp in summary_stats['models']:
            entropy_stats = summary_stats[f'{cp}_entropy_stats']
            f.write(f"  {cp}: mean={entropy_stats['mean']:.4f}, std={entropy_stats['std']:.4f}\n")
    
    print(f"\nSummary saved to: {summary_file}")

def create_top_tokens_csv(changes_df, timestamp):
    """Create a CSV file showing how top 20 most changed tokens evolved across checkpoints."""
    print("\nCreating CSV file for top 20 most changed tokens...")
    
    # Get the top 20 tokens with the largest absolute changes
    top_20_tokens = pd.concat([
        changes_df.nlargest(20, 'change'),
        changes_df.nsmallest(20, 'change')
    ]).drop_duplicates().nlargest(20, 'change')
    
    # Create a dictionary to store token probabilities across all checkpoints
    token_evolution = {}
    
    # Initialize with all unique tokens from the changes
    all_tokens = set()
    for _, row in changes_df.iterrows():
        all_tokens.add(row['token'])
    
    # For each token, get its probability in each checkpoint
    for token in all_tokens:
        token_evolution[token] = {
            'checkpoint_1_prob': 0.0,
            'checkpoint_2_prob': 0.0, 
            'checkpoint_3_prob': 0.0,
            'checkpoint_4_prob': 0.0
        }
    
    # Extract probabilities from the changes data
    # We need to reconstruct the full probability matrix from the changes
    checkpoint_pairs = changes_df['checkpoint_pair'].unique()
    
    # Start with checkpoint_1 probabilities (from checkpoint_1_to_checkpoint_2)
    cp1_to_cp2 = changes_df[changes_df['checkpoint_pair'] == 'checkpoint_1_to_checkpoint_2']
    for _, row in cp1_to_cp2.iterrows():
        token_evolution[row['token']]['checkpoint_1_prob'] = row['old_prob']
        token_evolution[row['token']]['checkpoint_2_prob'] = row['new_prob']
    
    # Get checkpoint_2_to_checkpoint_3 data
    cp2_to_cp3 = changes_df[changes_df['checkpoint_pair'] == 'checkpoint_2_to_checkpoint_3']
    for _, row in cp2_to_cp3.iterrows():
        token_evolution[row['token']]['checkpoint_3_prob'] = row['new_prob']
    
    # Get checkpoint_3_to_checkpoint_4 data  
    cp3_to_cp4 = changes_df[changes_df['checkpoint_pair'] == 'checkpoint_3_to_checkpoint_4']
    for _, row in cp3_to_cp4.iterrows():
        token_evolution[row['token']]['checkpoint_4_prob'] = row['new_prob']
    
    # Create DataFrame for top 20 tokens
    top_20_data = []
    for token in top_20_tokens['token']:
        if token in token_evolution:
            row_data = {'Token': token}
            row_data.update(token_evolution[token])
            top_20_data.append(row_data)
    
    top_20_df = pd.DataFrame(top_20_data)
    
    # Add change information
    top_20_df['total_change'] = top_20_df['checkpoint_4_prob'] - top_20_df['checkpoint_1_prob']
    top_20_df['max_change'] = top_20_df[['checkpoint_1_prob', 'checkpoint_2_prob', 'checkpoint_3_prob', 'checkpoint_4_prob']].max(axis=1) - top_20_df[['checkpoint_1_prob', 'checkpoint_2_prob', 'checkpoint_3_prob', 'checkpoint_4_prob']].min(axis=1)
    
    # Sort by total change magnitude
    top_20_df = top_20_df.sort_values('total_change', key=abs, ascending=False)
    
    # Save to CSV
    csv_filename = f'plots/top_20_tokens_evolution_{timestamp}.csv'
    top_20_df.to_csv(csv_filename, index=False)
    
    print(f"CSV file saved: {csv_filename}")
    print(f"Top 20 tokens with most probability changes:")
    print(top_20_df[['Token', 'checkpoint_1_prob', 'checkpoint_2_prob', 'checkpoint_3_prob', 'checkpoint_4_prob', 'total_change']].to_string(index=False))
    
    return top_20_df

def main():
    """Main function to run all checkpoint analysis visualizations."""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    try:
        summary_stats, avg_probs_df, changes_df, histogram_data, timestamp = load_checkpoint_data()
        print(f"Successfully loaded data for {len(summary_stats['models'])} checkpoints")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create CSV file for top 20 tokens
    top_20_df = create_top_tokens_csv(changes_df, timestamp)
    
    # Create all visualizations
    print("\nCreating checkpoint analysis visualizations...")
    
    plot_average_length_analysis(summary_stats)
    plot_entropy_changes_analysis(summary_stats)
    plot_token_probability_changes(changes_df)
    plot_most_changed_tokens(changes_df)
    plot_probability_histograms(histogram_data)
    
    # Create summary report
    create_summary_report(summary_stats, avg_probs_df, changes_df, timestamp)
    
    print(f"\nAll plots saved to 'plots/' directory")
    print("Checkpoint analysis complete!")

if __name__ == "__main__":
    main() 