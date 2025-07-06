import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_all_model_results(results_dir="results"):
    """Load results from all three models for comparison."""
    # Find the most recent timestamp from the files
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    
    # Extract timestamps from filenames
    timestamps = []
    for file in csv_files:
        if "_stats_" in file:
            # Extract timestamp from filename like "model_stats_20250706_000704.csv"
            parts = file.split("_stats_")
            if len(parts) > 1:
                timestamp_str = parts[1].replace(".csv", "")
                timestamps.append(timestamp_str)
    
    if not timestamps:
        raise FileNotFoundError("No model stats files found")
    
    latest_timestamp = max(timestamps)
    print(f"Using data from timestamp: {latest_timestamp}")
    
    # Define the three model names
    model_names = [
        "qwen_2_5_1_5b_instruct_reasoning_sft_checkpoint_1_of_10",
        "qwen_2_5_1_5b_instruct_reasoning_sft_checkpoint_2_of_10", 
        "qwen_2_5_1_5b_instruct_reasoning_sft_checkpoint_4_of_10"
    ]
    
    # Create mapping for clean checkpoint labels
    checkpoint_labels = {
        "qwen_2_5_1_5b_instruct_reasoning_sft_checkpoint_1_of_10": "Checkpoint 1",
        "qwen_2_5_1_5b_instruct_reasoning_sft_checkpoint_2_of_10": "Checkpoint 2", 
        "qwen_2_5_1_5b_instruct_reasoning_sft_checkpoint_4_of_10": "Checkpoint 4"
    }
    
    # Load data for each model
    all_models_data = {}
    all_models_additional = {}
    
    for model_name in model_names:
        # Load main stats file
        stats_file = os.path.join(results_dir, f"{model_name}_stats_{latest_timestamp}.csv")
        if os.path.exists(stats_file):
            df = pd.read_csv(stats_file)
            df['model'] = model_name  # Add model identifier
            all_models_data[model_name] = df
            print(f"Loaded {len(df)} tokens for {model_name}")
        else:
            print(f"Warning: Stats file not found for {model_name}")
            continue
        
        # Load additional files if they exist
        additional_data = {}
        high_entropy_file = os.path.join(results_dir, f"{model_name}_high_entropy_tokens_{latest_timestamp}.csv")
        low_entropy_file = os.path.join(results_dir, f"{model_name}_low_entropy_tokens_{latest_timestamp}.csv")
        confident_tokens_file = os.path.join(results_dir, f"{model_name}_confident_tokens_{latest_timestamp}.csv")
        
        if os.path.exists(high_entropy_file):
            additional_data['high_entropy'] = pd.read_csv(high_entropy_file)
            additional_data['high_entropy']['model'] = model_name
        if os.path.exists(low_entropy_file):
            additional_data['low_entropy'] = pd.read_csv(low_entropy_file)
            additional_data['low_entropy']['model'] = model_name
        if os.path.exists(confident_tokens_file):
            additional_data['confident_tokens'] = pd.read_csv(confident_tokens_file)
            additional_data['confident_tokens']['model'] = model_name
        
        all_models_additional[model_name] = additional_data
    
    # Combine all data into a single dataframe for easier plotting
    combined_df = pd.concat(all_models_data.values(), ignore_index=True)
    
    return combined_df, all_models_data, all_models_additional, latest_timestamp, checkpoint_labels

def plot_entropy_distribution(combined_df, all_models_data, timestamp, checkpoint_labels):
    """Plot 1: Distribution of entropy across all tokens for all models."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram with log scale - all models together
    for model_name, df in all_models_data.items():
        df['entropy'].hist(ax=ax1, bins=50, alpha=0.6, density=True, log=True, label=checkpoint_labels[model_name])
    ax1.set_title('Distribution of Token Entropy (Log Scale) - All Models')
    ax1.set_xlabel('Entropy')
    ax1.set_ylabel('Density (log scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot - all models together
    combined_df.boxplot(column='entropy', by='model', ax=ax2)
    ax2.set_title('Entropy Distribution Box Plot - All Models')
    ax2.set_ylabel('Entropy')
    ax2.set_xlabel('Model')
    ax2.grid(True, alpha=0.3)
    
    # Violin plot for better distribution comparison
    # Create a sample of data for violin plot to avoid memory issues
    sample_df = combined_df.sample(min(10000, len(combined_df)))
    sns.violinplot(data=sample_df, x='model', y='entropy', ax=ax3)
    ax3.set_title('Entropy Distribution Violin Plot - All Models')
    ax3.set_ylabel('Entropy')
    ax3.set_xlabel('Model')
    ax3.tick_params(axis='x', rotation=45)
    
    # KDE plot for smooth distribution comparison
    for model_name, df in all_models_data.items():
        df['entropy'].plot(kind='kde', ax=ax4, label=checkpoint_labels[model_name], alpha=0.7)
    ax4.set_title('Entropy Distribution KDE - All Models')
    ax4.set_xlabel('Entropy')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/entropy_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_probability_distribution(combined_df, all_models_data, timestamp, checkpoint_labels):
    """Plot 2: Distribution of probability of chosen tokens for all models."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram - all models together
    for model_name, df in all_models_data.items():
        df['probability_of_next_token'].hist(ax=ax1, bins=50, alpha=0.6, density=True, label=checkpoint_labels[model_name])
    ax1.set_title('Distribution of Token Probability - All Models')
    ax1.set_xlabel('Probability of Chosen Token')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot - all models together
    combined_df.boxplot(column='probability_of_next_token', by='model', ax=ax2)
    ax2.set_title('Token Probability Distribution Box Plot - All Models')
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Model')
    ax2.grid(True, alpha=0.3)
    
    # Violin plot for better distribution comparison
    # Create a sample of data for violin plot to avoid memory issues
    sample_df = combined_df.sample(min(10000, len(combined_df)))
    sns.violinplot(data=sample_df, x='model', y='probability_of_next_token', ax=ax3)
    ax3.set_title('Probability Distribution Violin Plot - All Models')
    ax3.set_ylabel('Probability')
    ax3.set_xlabel('Model')
    ax3.tick_params(axis='x', rotation=45)
    
    # KDE plot for smooth distribution comparison
    for model_name, df in all_models_data.items():
        df['probability_of_next_token'].plot(kind='kde', ax=ax4, label=checkpoint_labels[model_name], alpha=0.7)
    ax4.set_title('Probability Distribution KDE - All Models')
    ax4.set_xlabel('Probability')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/probability_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_entropy_vs_probability(combined_df, all_models_data, timestamp, checkpoint_labels):
    """Plot 3: Scatter plot of entropy vs probability for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Combined scatter plot with model colors
    colors = ['red', 'blue', 'green']
    for i, (model_name, df) in enumerate(all_models_data.items()):
        scatter = ax1.scatter(df['entropy'], df['probability_of_next_token'], 
                             alpha=0.6, s=20, c=df['token_position'], cmap='viridis',
                             label=checkpoint_labels[model_name])
    
    ax1.set_xlabel('Entropy')
    ax1.set_ylabel('Probability of Chosen Token')
    ax1.set_title('Entropy vs Probability of Chosen Token - All Models')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Separate scatter plots for each model
    for i, (model_name, df) in enumerate(all_models_data.items()):
        ax2.scatter(df['entropy'], df['probability_of_next_token'], 
                   alpha=0.6, s=20, label=checkpoint_labels[model_name], color=colors[i])
        
        # Add trend line for each model
        z = np.polyfit(df['entropy'], df['probability_of_next_token'], 1)
        p = np.poly1d(z)
        ax2.plot(df['entropy'], p(df['entropy']), color=colors[i], alpha=0.8, linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Entropy')
    ax2.set_ylabel('Probability of Chosen Token')
    ax2.set_title('Entropy vs Probability with Trend Lines - All Models')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/entropy_vs_probability_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_entropy_by_position(combined_df, all_models_data, timestamp, checkpoint_labels):
    """Plot 4: Entropy by token position in generation for all models."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Line plot of mean entropy by position for all models
    for model_name, df in all_models_data.items():
        position_entropy = df.groupby('token_position')['entropy'].agg(['mean', 'std', 'count']).reset_index()
        position_entropy = position_entropy[position_entropy['count'] >= 5]  # Filter for positions with enough samples
        
        ax1.plot(position_entropy['token_position'], position_entropy['mean'], 'o-', 
                linewidth=2, markersize=4, label=checkpoint_labels[model_name])
        ax1.fill_between(position_entropy['token_position'], 
                         position_entropy['mean'] - position_entropy['std'],
                         position_entropy['mean'] + position_entropy['std'], 
                         alpha=0.3)
    
    ax1.set_xlabel('Token Position in Generation')
    ax1.set_ylabel('Mean Entropy')
    ax1.set_title('Mean Entropy by Token Position - All Models')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Heatmap of entropy by position and model
    # Create a sample for heatmap to avoid memory issues
    sample_df = combined_df.sample(min(5000, len(combined_df)))
    pivot_data = sample_df.pivot_table(values='entropy', index='model', columns='token_position', aggfunc='mean')
    sns.heatmap(pivot_data, ax=ax2, cmap='viridis', cbar_kws={'label': 'Entropy'})
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Model')
    ax2.set_title('Entropy Heatmap: Model vs Token Position')
    
    # Box plot of entropy by position ranges
    # Create a sample for boxplot to avoid memory issues
    sample_df = combined_df.sample(min(5000, len(combined_df)))
    sample_df['position_bin'] = pd.cut(sample_df['token_position'], bins=10, labels=False)
    sns.boxplot(data=sample_df, x='position_bin', y='entropy', hue='model', ax=ax3)
    ax3.set_xlabel('Token Position Bin')
    ax3.set_ylabel('Entropy')
    ax3.set_title('Entropy Distribution by Position Bins - All Models')
    ax3.legend(title='Model')
    
    # Entropy trend over positions
    for model_name, df in all_models_data.items():
        trend_data = df.groupby('token_position')['entropy'].mean().rolling(window=5, min_periods=1).mean()
        ax4.plot(trend_data.index, trend_data.values, label=checkpoint_labels[model_name], linewidth=2)
    
    ax4.set_xlabel('Token Position')
    ax4.set_ylabel('Smoothed Mean Entropy')
    ax4.set_title('Smoothed Entropy Trend by Position - All Models')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/entropy_by_position_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_token_analysis(combined_df, all_models_data, timestamp, checkpoint_labels):
    """Plot 5: Analysis of specific token types for all models."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Analyze tokens by length for all models
    for model_name, df in all_models_data.items():
        df['token_length'] = df['current_token'].str.len()
        length_entropy = df.groupby('token_length')['entropy'].mean().reset_index()
        ax1.scatter(length_entropy['token_length'], length_entropy['entropy'], 
                   alpha=0.7, label=checkpoint_labels[model_name])
    
    ax1.set_xlabel('Token Length')
    ax1.set_ylabel('Mean Entropy')
    ax1.set_title('Entropy vs Token Length - All Models')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Analyze tokens by character type for all models
    token_types = ['has_digit', 'has_letter', 'has_special']
    type_labels = ['Digits', 'Letters', 'Special']
    
    # Create grouped bar chart for token characteristics
    type_entropy_data = []
    for model_name in all_models_data.keys():
        df = all_models_data[model_name]
        for token_type in token_types:
            df[token_type] = df['current_token'].str.contains(
                r'\d' if token_type == 'has_digit' else 
                r'[a-zA-Z]' if token_type == 'has_letter' else 
                r'[^a-zA-Z0-9\s]', na=False)
            
            type_entropy_data.append({
                'model': checkpoint_labels[model_name],
                'type': token_type,
                'has_feature': True,
                'mean_entropy': df[df[token_type]]['entropy'].mean()
            })
            type_entropy_data.append({
                'model': checkpoint_labels[model_name],
                'type': token_type,
                'has_feature': False,
                'mean_entropy': df[~df[token_type]]['entropy'].mean()
            })
    
    type_entropy_df = pd.DataFrame(type_entropy_data)
    sns.barplot(data=type_entropy_df, x='type', y='mean_entropy', hue='model', ax=ax2)
    ax2.set_ylabel('Mean Entropy')
    ax2.set_title('Entropy by Token Characteristics - All Models')
    ax2.legend(title='Model')
    ax2.grid(True, alpha=0.3)
    
    # Most common tokens and their entropy across models
    common_tokens_data = []
    for model_name, df in all_models_data.items():
        token_entropy = df.groupby('current_token')['entropy'].agg(['mean', 'count']).reset_index()
        token_entropy = token_entropy[token_entropy['count'] >= 5].sort_values('count', ascending=False).head(10)
        token_entropy['model'] = checkpoint_labels[model_name]
        common_tokens_data.append(token_entropy)
    
    if common_tokens_data:
        common_tokens_df = pd.concat(common_tokens_data, ignore_index=True)
        # Use a simpler approach for the bar plot
        for model_name in common_tokens_df['model'].unique():
            model_data = common_tokens_df[common_tokens_df['model'] == model_name]
            ax3.barh(range(len(model_data)), model_data['mean'], alpha=0.7, label=model_name)
        
        ax3.set_xlabel('Mean Entropy')
        ax3.set_ylabel('Token Index')
        ax3.set_title('Mean Entropy for Most Common Tokens - All Models')
        ax3.legend(title='Model')
        ax3.grid(True, alpha=0.3)
    
    # Entropy distribution by prompt across models
    for model_name, df in all_models_data.items():
        prompt_entropy = df.groupby('prompt_id')['entropy'].agg(['mean', 'std']).reset_index()
        ax4.scatter(prompt_entropy['mean'], prompt_entropy['std'], 
                   alpha=0.7, label=checkpoint_labels[model_name])
    
    ax4.set_xlabel('Mean Entropy per Prompt')
    ax4.set_ylabel('Std Entropy per Prompt')
    ax4.set_title('Entropy Statistics by Prompt - All Models')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/token_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_quantile_analysis(combined_df, all_models_data, timestamp, checkpoint_labels):
    """Plot 6: Quantile analysis for all models."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Create quantiles for entropy for each model
    n_qcuts = 10
    for model_name, df in all_models_data.items():
        df['entropy_qcut'] = pd.qcut(df['entropy'], n_qcuts, labels=False, duplicates='drop')
    
    # Quantile statistics for each model
    for model_name, df in all_models_data.items():
        qcut_stats = df.groupby('entropy_qcut').agg({
            'entropy': ['mean', 'std', 'count'],
            'probability_of_next_token': ['mean', 'std']
        }).reset_index()
        
        # Plot mean entropy by quantile
        ax1.plot(qcut_stats['entropy_qcut'], qcut_stats[('entropy', 'mean')], 
                'o-', linewidth=2, label=checkpoint_labels[model_name])
        
        # Plot mean probability by quantile
        ax2.plot(qcut_stats['entropy_qcut'], qcut_stats[('probability_of_next_token', 'mean')], 
                'o-', linewidth=2, label=checkpoint_labels[model_name])
    
    ax1.set_xlabel('Entropy Quantile')
    ax1.set_ylabel('Mean Entropy')
    ax1.set_title('Mean Entropy by Quantile - All Models')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Entropy Quantile')
    ax2.set_ylabel('Mean Probability')
    ax2.set_title('Mean Probability by Entropy Quantile - All Models')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Sample tokens from different quantiles for each model
    ax3.axis('off')
    sample_text = "Sample tokens from different entropy quantiles:\n\n"
    
    for model_name, df in all_models_data.items():
        sample_text += f"\n{checkpoint_labels[model_name]}:\n"
        for qcut in [0, n_qcuts//4, n_qcuts//2, 3*n_qcuts//4, n_qcuts-1]:
            if qcut in df['entropy_qcut'].values:
                sample = df[df['entropy_qcut'] == qcut].sample(min(2, len(df[df['entropy_qcut'] == qcut])))
                sample_text += f"  Q{qcut}: "
                for _, row in sample.head(2).iterrows():
                    sample_text += f"'{row['current_token']}'->'{row['next_token']}'({row['entropy']:.2f}) "
                sample_text += "\n"
    
    ax3.text(0.05, 0.95, sample_text, transform=ax3.transAxes, fontsize=8, 
             verticalalignment='top', fontfamily='monospace')
    
    # Distribution comparison for high vs low entropy across models
    for model_name, df in all_models_data.items():
        low_entropy = df[df['entropy_qcut'] <= n_qcuts//3]['entropy']
        high_entropy = df[df['entropy_qcut'] >= 2*n_qcuts//3]['entropy']
        
        ax4.hist(low_entropy, bins=30, alpha=0.5, label=f'{checkpoint_labels[model_name]} Low', density=True)
        ax4.hist(high_entropy, bins=30, alpha=0.5, label=f'{checkpoint_labels[model_name]} High', density=True)
    
    ax4.set_xlabel('Entropy')
    ax4.set_ylabel('Density')
    ax4.set_title('Entropy Distribution: Low vs High Quantiles - All Models')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/quantile_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_interesting_tokens(all_models_additional, timestamp, checkpoint_labels):
    """Plot 7: Analysis of interesting tokens (high/low entropy, confident) for all models."""
    if not all_models_additional:
        print("No additional data files found for interesting tokens analysis")
        return
    
    # Determine how many interesting token types are present
    token_types = []
    for t in ['high_entropy', 'low_entropy', 'confident_tokens']:
        for model_data in all_models_additional.values():
            if t in model_data:
                token_types.append(t)
                break
    n_plots = len(token_types)
    if n_plots == 0:
        print("No interesting token data to plot.")
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    for plot_idx, token_type in enumerate(token_types):
        data_list = []
        for model_name, additional_data in all_models_additional.items():
            if token_type in additional_data:
                df = additional_data[token_type].copy()
                df['label'] = checkpoint_labels[model_name]
                data_list.append(df)
        if not data_list:
            continue
        combined = pd.concat(data_list, ignore_index=True)
        if token_type == 'confident_tokens':
            value_col = 'probability_of_next_token'
            color = 'green'
            xlabel = 'Probability'
            title = 'Top Confident'
        else:
            value_col = 'entropy'
            color = 'red' if token_type == 'high_entropy' else 'blue'
            xlabel = 'Entropy'
            title = 'Top High Entropy' if token_type == 'high_entropy' else 'Top Low Entropy'
        # Plot top 5 from each model
        top_tokens = combined.groupby('label').head(5)
        for i, label in enumerate(top_tokens['label'].unique()):
            model_tokens = top_tokens[top_tokens['label'] == label]
            axes[plot_idx].barh(range(len(model_tokens)), model_tokens[value_col], alpha=0.7, color=color, label=label)
        axes[plot_idx].set_yticks(range(len(model_tokens)))
        axes[plot_idx].set_yticklabels([f"'{row['current_token']}'->'{row['next_token']}'" for _, row in model_tokens.iterrows()], fontsize=6)
        axes[plot_idx].set_xlabel(xlabel)
        axes[plot_idx].set_title(title)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/interesting_tokens_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_prompt_analysis(combined_df, all_models_data, timestamp, checkpoint_labels):
    """Plot 8: Analysis across different prompts for all models."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Entropy statistics by prompt for each model
    for model_name, df in all_models_data.items():
        prompt_stats = df.groupby('prompt_id').agg({
            'entropy': ['mean', 'std', 'min', 'max'],
            'probability_of_next_token': ['mean', 'std'],
            'token_position': 'count'
        }).reset_index()
        
        # Flatten column names
        prompt_stats.columns = ['prompt_id', 'mean_entropy', 'std_entropy', 'min_entropy', 'max_entropy',
                               'mean_prob', 'std_prob', 'token_count']
        
        # Mean entropy by prompt
        ax1.plot(prompt_stats['prompt_id'], prompt_stats['mean_entropy'], 
                'o-', label=checkpoint_labels[model_name], alpha=0.7)
    
    ax1.set_xlabel('Prompt ID')
    ax1.set_ylabel('Mean Entropy')
    ax1.set_title('Mean Entropy by Prompt - All Models')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Entropy range by prompt for each model
    for model_name, df in all_models_data.items():
        prompt_stats = df.groupby('prompt_id').agg({
            'entropy': ['mean', 'min', 'max'],
            'token_position': 'count'
        }).reset_index()
        prompt_stats.columns = ['prompt_id', 'mean_entropy', 'min_entropy', 'max_entropy', 'token_count']
        
        ax2.fill_between(prompt_stats['prompt_id'], 
                         prompt_stats['min_entropy'], 
                         prompt_stats['max_entropy'], 
                         alpha=0.3, label=f'{checkpoint_labels[model_name]} Range')
        ax2.plot(prompt_stats['prompt_id'], prompt_stats['mean_entropy'], 
                'o-', linewidth=2, label=f'{checkpoint_labels[model_name]} Mean', alpha=0.7)
    
    ax2.set_xlabel('Prompt ID')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Entropy Range by Prompt - All Models')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Token count vs mean entropy for each model
    for model_name, df in all_models_data.items():
        prompt_stats = df.groupby('prompt_id').agg({
            'entropy': 'mean',
            'token_position': 'count'
        }).reset_index()
        prompt_stats.columns = ['prompt_id', 'mean_entropy', 'token_count']
        
        ax3.scatter(prompt_stats['token_count'], prompt_stats['mean_entropy'], 
                   alpha=0.7, label=checkpoint_labels[model_name])
    
    ax3.set_xlabel('Number of Tokens Generated')
    ax3.set_ylabel('Mean Entropy')
    ax3.set_title('Mean Entropy vs Generation Length - All Models')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Correlation between entropy and probability for each model
    for model_name, df in all_models_data.items():
        prompt_stats = df.groupby('prompt_id').agg({
            'entropy': 'mean',
            'probability_of_next_token': 'mean'
        }).reset_index()
        prompt_stats.columns = ['prompt_id', 'mean_entropy', 'mean_prob']
        
        prompt_corr = prompt_stats['mean_entropy'].corr(prompt_stats['mean_prob'])
        ax4.scatter(prompt_stats['mean_entropy'], prompt_stats['mean_prob'], 
                   alpha=0.7, label=f'{checkpoint_labels[model_name]} (corr: {prompt_corr:.3f})')
    
    ax4.set_xlabel('Mean Entropy')
    ax4.set_ylabel('Mean Probability')
    ax4.set_title('Mean Entropy vs Mean Probability - All Models')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/prompt_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics(combined_df, all_models_data, all_models_additional, timestamp, checkpoint_labels):
    """Create a comprehensive summary of the analysis for all models."""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS SUMMARY - {timestamp}")
    print(f"{'='*80}")
    
    print(f"\nDataset Overview:")
    print(f"  Total tokens analyzed: {len(combined_df):,}")
    print(f"  Total prompts processed: {combined_df['prompt_id'].nunique()}")
    print(f"  Models analyzed: {len(all_models_data)}")
    
    for model_name, df in all_models_data.items():
        print(f"\n{checkpoint_labels[model_name]} Statistics:")
        print(f"  Total tokens: {len(df):,}")
        print(f"  Mean entropy: {df['entropy'].mean():.4f}")
        print(f"  Median entropy: {df['entropy'].median():.4f}")
        print(f"  Std entropy: {df['entropy'].std():.4f}")
        print(f"  Mean probability: {df['probability_of_next_token'].mean():.4f}")
        print(f"  Median probability: {df['probability_of_next_token'].median():.4f}")
        
        # Correlation analysis
        corr = df['entropy'].corr(df['probability_of_next_token'])
        pos_corr = df['token_position'].corr(df['entropy'])
        print(f"  Entropy vs Probability correlation: {corr:.4f}")
        print(f"  Token position vs Entropy correlation: {pos_corr:.4f}")
    
    # Model comparison
    print(f"\nModel Comparison:")
    comparison_data = []
    for model_name, df in all_models_data.items():
        comparison_data.append({
            'model': checkpoint_labels[model_name],
            'mean_entropy': df['entropy'].mean(),
            'std_entropy': df['entropy'].std(),
            'mean_probability': df['probability_of_next_token'].mean(),
            'std_probability': df['probability_of_next_token'].std(),
            'total_tokens': len(df)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    if all_models_additional:
        print(f"\nInteresting Tokens Summary:")
        for model_name, additional_data in all_models_additional.items():
            print(f"  {checkpoint_labels[model_name]}:")
            for key, data in additional_data.items():
                print(f"    {key}: {len(data)} tokens")
    
    # Save summary to file
    summary_file = f'plots/analysis_summary_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write(f"COMPREHENSIVE ANALYSIS SUMMARY - {timestamp}\n")
        f.write("="*80 + "\n")
        f.write(f"\nDataset Overview:\n")
        f.write(f"  Total tokens analyzed: {len(combined_df):,}\n")
        f.write(f"  Total prompts processed: {combined_df['prompt_id'].nunique()}\n")
        f.write(f"  Models analyzed: {len(all_models_data)}\n")
        
        for model_name, df in all_models_data.items():
            f.write(f"\n{checkpoint_labels[model_name]} Statistics:\n")
            f.write(f"  Total tokens: {len(df):,}\n")
            f.write(f"  Mean entropy: {df['entropy'].mean():.4f}\n")
            f.write(f"  Mean probability: {df['probability_of_next_token'].mean():.4f}\n")
        
        f.write(f"\nModel Comparison:\n")
        f.write(comparison_df.to_string(index=False))

def main():
    """Main function to run all visualizations for all models."""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load data for all models
    try:
        combined_df, all_models_data, all_models_additional, timestamp, checkpoint_labels = load_all_model_results()
        print(f"Loaded data for {len(all_models_data)} models:")
        for model_name, df in all_models_data.items():
            print(f"  {checkpoint_labels[model_name]}: {len(df)} tokens from {df['prompt_id'].nunique()} prompts")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create all plots
    print("\nCreating visualizations for all models...")
    
    plot_entropy_distribution(combined_df, all_models_data, timestamp, checkpoint_labels)
    plot_probability_distribution(combined_df, all_models_data, timestamp, checkpoint_labels)
    plot_entropy_vs_probability(combined_df, all_models_data, timestamp, checkpoint_labels)
    plot_entropy_by_position(combined_df, all_models_data, timestamp, checkpoint_labels)
    plot_token_analysis(combined_df, all_models_data, timestamp, checkpoint_labels)
    plot_quantile_analysis(combined_df, all_models_data, timestamp, checkpoint_labels)
    plot_interesting_tokens(all_models_additional, timestamp, checkpoint_labels)
    plot_prompt_analysis(combined_df, all_models_data, timestamp, checkpoint_labels)
    
    # Create summary
    create_summary_statistics(combined_df, all_models_data, all_models_additional, timestamp, checkpoint_labels)
    
    print(f"\nAll plots saved to 'plots/' directory with timestamp {timestamp}")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 